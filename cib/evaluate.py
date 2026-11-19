"""One evaluation run: background split + one level of synthetic campaigns (+ benign-LLM controls) -> scores of the
method, its ablations and the baselines, all thresholded at the same organic false-alarm rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cib import baselines, fuse, gnn, metrics, pipeline
from cib.generate import inject
from cib.layers import style


def layer_score(edges: pd.DataFrame, agg: str = "top3") -> pd.Series:
    """Account score from one layer: sum of -log10 p over its candidate edges."""
    return pipeline.evidence({"x": edges}, agg=agg)


def raw_layer_score(res: pipeline.Result, key: str, msgs: pd.DataFrame) -> pd.Series:
    return baselines.raw_pairs(msgs, res.raw_pairs[key]) if key in res.raw_pairs else pd.Series(dtype=float)


def node_features(msgs: pd.DataFrame, machine_prob: pd.Series | None) -> pd.DataFrame:
    """Text-only node features (no activity profile: see the artifact audit decision of 2026-11-18)."""
    m = msgs[msgs.kind.ne("post") & msgs.text.str.len().gt(0)]
    hand = style.handcrafted(m.text).groupby(m.account_id.to_numpy()).mean()
    if machine_prob is not None:
        p = machine_prob.reindex(m.msg_id).to_numpy()
        hand["m_bar"] = pd.Series(p, index=m.account_id.to_numpy()).groupby(level=0).mean()
    return hand.fillna(0.0)


def labels(msgs: pd.DataFrame) -> pd.DataFrame:
    g = msgs[msgs.kind != "post"].groupby("account_id")
    # GroupBy has a .level attribute of its own, so columns are accessed by key
    out = pd.DataFrame({"positive": g["campaign_id"].agg(lambda s: s.notna().any()),
                        "benign": g["level"].agg(lambda s: s.isin(["benign_llm", "benign_hot"]).any()),
                        "benign_hot": g["level"].agg(lambda s: (s == "benign_hot").any()),
                        "injected": g["is_injected"].any()})
    out["organic"] = ~out.injected
    out["campaign"] = g["campaign_id"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
    return out


H3_DETECTORS = ("full", "full_noMY", "layer_T", "layer_S", "raw_T", "raw_S")


def score_table(scores: dict[str, pd.Series], lab: pd.DataFrame, fprs=(0.001, 0.01),
                clusters: dict[str, pd.DataFrame] | None = None, msgs: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        s = s.reindex(lab.index, fill_value=0.0).fillna(0.0)
        r = {"detector": name, "n_pos": int(lab.positive.sum()), "n_benign": int(lab.benign.sum()),
             "auprc": metrics.auprc(s, lab.positive, lab.organic | lab.benign)}
        for f in fprs:
            rec, thr = metrics.recall_at_fpr(s, lab.positive, lab.organic, f)
            r[f"recall@{f}"] = rec
            r[f"benign_flag@{f}"] = metrics.flagged_rate(s, lab.index[lab.benign], thr) if lab.benign.any() else np.nan
            if "benign_hot" in lab and lab.benign_hot.any():
                r[f"benign_hot_flag@{f}"] = metrics.flagged_rate(s, lab.index[lab.benign_hot], thr)
            r[f"organic_flag@{f}"] = metrics.flagged_rate(s, lab.index[lab.organic], thr)
        if msgs is not None and name in H3_DETECTORS:
            _, thr = metrics.recall_at_fpr(s, lab.positive, lab.organic, 0.01)
            h3 = metrics.burst_flag_ratio(s, msgs, lab.index[lab.organic], thr)
            r["h3_flag_burst@0.01"], r["h3_flag_other@0.01"], r["h3_ratio@0.01"] = (h3["flag_burst"],
                                                                                   h3["flag_other"], h3["ratio"])
        if clusters and name in clusters:
            f1 = metrics.best_cluster_f1(clusters[name], lab.campaign)
            r["f1_star_mean"] = float(f1.mean()) if len(f1) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def run_once(bg: pd.DataFrame, syn: pd.DataFrame, cfg: dict, *, emb: np.ndarray, row_of: pd.Series,
             machine_prob: pd.Series | None, seed: int = 0, log=print, extra_scores: dict | None = None
             ) -> tuple[pd.DataFrame, pipeline.Result, pd.DataFrame]:
    msgs = inject.inject(bg, syn)
    res = pipeline.run(msgs, cfg, emb=emb, row_of=row_of, machine_prob=machine_prob, seed=seed, log=log)
    lab = labels(msgs)
    scores = {"full": res.score}
    for k in ("T", "S", "L"):
        if k in res.layers:
            scores[f"layer_{k}"] = layer_score(res.layers[k], res.agg)
    scores["full_noMY"] = pipeline.evidence(res.layers, ("T", "S", "L"), agg=res.agg)
    if "Y" in res.layers:
        scores["full_noM"] = pipeline.evidence(res.layers, ("T", "S", "L", "Y"), agg=res.agg)
    scores["raw_T"] = raw_layer_score(res, "T", msgs)
    scores["raw_S"] = raw_layer_score(res, "S", msgs)
    scores["copypaste"] = baselines.copypaste(msgs)
    if machine_prob is not None:
        scores["account_machine"] = baselines.account_machine(msgs[msgs.kind != "post"], machine_prob)
    scores.update(extra_scores or {})
    table = score_table(scores, lab, clusters={"full": res.clusters}, msgs=msgs)
    return table, res, lab


def gnn_scores(train: tuple[pipeline.Result, pd.DataFrame, pd.DataFrame], test: tuple[pipeline.Result, pd.DataFrame,
               pd.DataFrame], layer_cols: list[str], cfg: dict, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Train on one run (its accounts and labels), predict on another; returns account probability and groups.

    Each tuple: (pipeline result, labels, node features)."""
    (r_tr, lab_tr, nf_tr), (r_te, lab_te, nf_te) = train, test
    g = cfg["gnn"]
    d_tr, names_tr = gnn.build_graph(r_tr.fused.assign(w=r_tr.fused.w), layer_cols, nf_tr)
    y = lab_tr.positive.reindex(names_tr).fillna(False).to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    train_mask = rng.random(len(names_tr)) < 0.8
    model = gnn.train(d_tr, y, train_mask, val_mask=~train_mask, seed=seed, hidden=g["hidden"], layers=g["layers"],
                      dropout=g["dropout"], conv=g["conv"])
    d_te, names_te = gnn.build_graph(r_te.fused, layer_cols, nf_te.reindex(columns=nf_tr.columns, fill_value=0))
    prob = gnn.predict(model, d_te, names_te)
    org = lab_te.index[lab_te.organic]
    theta = metrics.threshold_at_fpr(prob.reindex(org, fill_value=0).to_numpy(), 0.001)
    groups = gnn.groups(r_te.fused, prob, theta=theta, resolution=cfg["fusion"]["leiden_resolution"],
                        min_cluster=cfg["fusion"]["min_cluster"], seed=seed)
    return prob, groups


def per_campaign_recall(scores: dict[str, pd.Series], lab: pd.DataFrame, fpr: float,
                        group: pd.Series | None = None) -> pd.DataFrame:
    """Recall of each campaign (optionally split by `group`, e.g. generator of the account) at the organic-FPR
    threshold of each detector. Long format: detector, campaign, [group], n, recall."""
    rows = []
    pos = lab[lab.positive]
    for name, s in scores.items():
        s = s.reindex(lab.index, fill_value=0.0).fillna(0.0)
        _, thr = metrics.recall_at_fpr(s, lab.positive, lab.organic, fpr)
        hit = s.reindex(pos.index) > thr
        keys = [pos.campaign] if group is None else [pos.campaign, group.reindex(pos.index)]
        for k, h in hit.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({"detector": name, "campaign": k[0], **({"group": k[1]} if group is not None else {}),
                         "n": int(len(h)), "recall": float(h.mean()), "fpr": fpr})
    return pd.DataFrame(rows)
