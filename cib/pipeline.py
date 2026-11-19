"""End-to-end detection on one message table: layers -> BPAP significance -> fusion -> Leiden -> account scores.

Returns everything needed by the GNN and by evaluation: per-layer edge tables (account ids), the fused wide edge
table and the no-GNN clusters/scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cib import fuse, null
from cib.layers import cotarget, lexical, semantic, style
from cib.layers.pairs import account_codes


@dataclass
class Result:
    layers: dict[str, pd.DataFrame]
    fused: pd.DataFrame
    clusters: pd.DataFrame
    raw_pairs: dict[str, object] = field(default_factory=dict)

    agg: str = "top3"

    @property
    def score(self) -> pd.Series:
        """Account score from pair evidence summed over all layers; the organic false-alarm rate is fixed later by
        thresholding this score, BH significance is only used to build the graph for communities."""
        return evidence(self.layers, agg=self.agg)


def pair_evidence(layers: dict[str, pd.DataFrame], keys: tuple[str, ...] | None = None) -> pd.DataFrame:
    """a, b, ev: sum over layers of -log10 p for every pair present in at least one selected layer."""
    parts = []
    for k, e in layers.items():
        if (keys is not None and k not in keys) or e.empty:
            continue
        a, b = e.a.astype(str).to_numpy(), e.b.astype(str).to_numpy()
        parts.append(pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b),
                                   "ev": -np.log10(np.clip(e.p.to_numpy(float), 1e-300, 1))}))
    if not parts:
        return pd.DataFrame(columns=["a", "b", "ev"])
    return pd.concat(parts).groupby(["a", "b"], as_index=False).ev.sum()


def evidence(layers: dict[str, pd.DataFrame], keys: tuple[str, ...] | None = None, agg: str = "top3") -> pd.Series:
    """Account score from pair evidence. `max`: strongest partner; `top3`: sum of the three strongest; `sum`: all
    pairs (rewards hubs with many weak links); `excess`: sum of max(0, ev - 2)."""
    pe = pair_evidence(layers, keys)
    if pe.empty:
        return pd.Series(dtype=float)
    long = pd.concat([pe.rename(columns={"a": "acc"})[["acc", "ev"]], pe.rename(columns={"b": "acc"})[["acc", "ev"]]])
    g = long.groupby("acc").ev
    if agg == "max":
        return g.max()
    if agg == "top3":
        return g.apply(lambda x: x.nlargest(3).sum())
    if agg == "excess":
        return long.assign(ev=(long.ev - 2).clip(lower=0)).groupby("acc").ev.sum()
    return g.sum()


def _named(e: pd.DataFrame, names: pd.Index) -> pd.DataFrame:
    return e.assign(a=names[e.u.to_numpy()], b=names[e.v.to_numpy()]) if len(e) else e.assign(a=[], b=[])


def run(msgs: pd.DataFrame, cfg: dict, *, emb: np.ndarray | None = None, row_of: pd.Series | None = None,
        machine_prob: pd.Series | None = None, global_null: bool = False, seed: int = 0, log=print) -> Result:
    msgs = msgs.reset_index(drop=True)
    acc, names = account_codes(msgs.account_id)
    movable = msgs.kind.ne("post").to_numpy()
    nc = cfg["null_model"]
    bucket = null.buckets(msgs, slot_s={"community_hour": 3600, "community_3h": 10800,
                                         "community_day": 86400}[nc["bucket"]])
    L = cfg["layers"]
    pairs = {"T": cotarget.message_pairs(msgs, L["cotarget"]["window_s"]),
             "L": lexical.message_pairs(msgs, L["lexical"]["window_s"], L["lexical"]["min_jaccard"],
                                        L["lexical"]["shingle"], L["semantic"]["min_chars"])}
    if emb is not None:
        pairs["S"] = semantic.message_pairs(msgs, emb, row_of, L["semantic"]["window_s"], L["semantic"]["min_cos"])
    layers = {}
    for k, p in pairs.items():
        e = null.edge_significance(p, acc, len(names), bucket, movable, permutations=nc["permutations"],
                                   min_obs=nc["min_obs"], fdr_q=nc["fdr_q"], seed=seed, global_null=global_null)
        layers[k] = _named(e, names)
        log(f"layer {k}: {len(p)} message pairs, {len(e)} candidate edges, {int(e.sig.sum()) if len(e) else 0} sig")
    sync_pairs = pd.concat([layers[k][["a", "b"]] for k in ("T", "S") if k in layers and len(layers[k])]
                           ).drop_duplicates() if any(len(layers.get(k, [])) for k in ("T", "S")) else None
    if L["style"]["enabled"] and sync_pairs is not None:
        v, accs = style.account_vectors(msgs, L["style"]["min_messages"], seed=seed)
        layers["Y"] = style.pair_edges(v, accs, sync_pairs, seed=seed)
        log(f"layer Y: {len(layers['Y'])} synchronised pairs scored")
    if machine_prob is not None:
        layers["M"] = machine_edges(layers, msgs, machine_prob, cfg)
    f = fuse.fuse({k: v for k, v in layers.items()}, method=cfg["fusion"]["method"], fdr_q=nc["fdr_q"])
    cl = fuse.leiden(f, resolution=cfg["fusion"]["leiden_resolution"], min_cluster=cfg["fusion"]["min_cluster"],
                     seed=seed)
    return Result(layers=layers, fused=f, clusters=cl, raw_pairs=pairs, agg=cfg["fusion"].get("account_agg", "top3"))


def machine_edges(layers: dict[str, pd.DataFrame], msgs: pd.DataFrame, prob: pd.Series, cfg: dict) -> pd.DataFrame:
    """M only reinforces pairs already linked in T or S: p = 1 - min(m_u, m_v), significant if both accounts are
    above the threshold. On its own, M would just find LLM users, which H5 forbids."""
    from cib.layers.machine import account_mean
    m = account_mean(msgs, prob)
    base = pd.concat([layers[k][["a", "b"]] for k in ("T", "S") if k in layers and len(layers[k])])
    if base.empty:
        return pd.DataFrame(columns=["a", "b", "p", "sig"])
    base = base.drop_duplicates()
    mm = np.minimum(m.reindex(base.a).to_numpy(), m.reindex(base.b).to_numpy())
    thr = cfg["layers"]["machine"].get("min_prob", 0.5)
    return base.assign(p=np.clip(1 - np.nan_to_num(mm, nan=0.0), 1e-6, 1), sig=np.nan_to_num(mm, nan=0) >= thr)
