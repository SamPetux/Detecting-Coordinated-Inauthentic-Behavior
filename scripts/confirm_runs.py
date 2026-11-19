"""Confirmatory runs (docs/06_protocol.md): L0-L5 x seeds 11-13 on the confirm split, frozen configuration.

Per run: summary table (all detectors) and per-campaign recall (by account generator), resumable by output file.
M* = pooled graph-only GNN trained on exploration (results/models/gnn_pooled.pt); M0 = calibrated graph, top3.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cib import baselines, embed, evaluate, gnn
from cib.generate import inject

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
SEEDS = (11, 12, 13)
OUT = Path("results/confirm")


def load_gnn() -> tuple[gnn.EdgeGNN, list[str]]:
    meta = json.loads(Path("results/models/gnn_pooled.json").read_text())
    model = gnn.EdgeGNN(0, meta["edge_dim"], hidden=meta["hidden"], layers=meta["layers"], dropout=meta["dropout"],
                        conv=meta["conv"])
    model.load_state_dict(torch.load("results/models/gnn_pooled.pt"))
    return model.eval(), meta["edge_cols"]


def main() -> None:
    cfg = yaml.safe_load(open("configs/confirm.yaml"))
    assert cfg.get("frozen"), "protocol not frozen"
    OUT.mkdir(parents=True, exist_ok=True)
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[bg.split == "confirm"].reset_index(drop=True)
    syn = pd.read_parquet("data/synthetic/confirm/messages.parquet")
    plan = pd.read_parquet("data/synthetic/confirm/plan.parquet")
    gen_of = plan.drop_duplicates("account_id").set_index("account_id").model
    xb, ib = embed.embed_messages(bg, "intfloat/multilingual-e5-small", cache_dir="data/interim/emb/vk_confirm")
    xs, is_ = embed.embed_messages(syn, "intfloat/multilingual-e5-small", cache_dir="data/interim/emb/syn_confirm")
    emb = np.vstack([xb, xs])
    row_of = pd.Series(np.arange(len(emb)), index=ib.append(is_))
    ok = pd.concat([bg, syn]).set_index("msg_id").text.map(lambda t: len(embed.clean_for_similarity(t)) >= 20)
    row_of = row_of[row_of.index.isin(ok.index[ok])]
    kw = dict(preprocess=baselines.luceri_clean, keep=lambda t: len(t.split()) > 4)
    lb, lib = embed.embed_messages(bg, baselines.LUCERI_MODEL, cache_dir="data/interim/emb/vk_confirm_luceri", **kw)
    ls_, lis = embed.embed_messages(syn, baselines.LUCERI_MODEL, cache_dir="data/interim/emb/syn_confirm_luceri", **kw)
    luceri = (np.vstack([lb, ls_]), pd.Series(np.arange(len(lb) + len(ls_)), index=lib.append(lis)))
    prob = pd.read_parquet("data/interim/machine_prob_confirm.parquet").p
    model, cols = load_gnn()
    cid = syn.campaign_id.fillna("")
    controls = syn.level.isin(["benign_llm", "benign_hot"])
    for seed in SEEDS:
        for level in LEVELS:
            f = OUT / f"{level}_s{seed}.csv"
            if f.exists():
                continue
            sub = syn[cid.str.contains(f"-{level}-s{seed}") | controls]
            msgs = inject.inject(bg, sub)
            extra = {"luceri": baselines.luceri_text_similarity(msgs, *luceri)}
            table, res, lab = evaluate.run_once(bg, sub, copy.deepcopy(cfg), emb=emb, row_of=row_of,
                                                machine_prob=prob, log=lambda s: print(f"  {s}", flush=True),
                                                extra_scores=extra)
            d, names = gnn.build_graph(res.fused, cols, pd.DataFrame(index=lab.index))
            p_star = gnn.predict(model, d, names)
            star = evaluate.score_table({"M_star": p_star}, lab, msgs=msgs)
            table = pd.concat([star, table], ignore_index=True)
            table.insert(0, "level", level)
            table.insert(1, "seed", seed)
            scores = {"M_star": p_star, "M0": res.score, "layer_T": evaluate.layer_score(res.layers["T"], res.agg),
                      "layer_S": evaluate.layer_score(res.layers["S"], res.agg),
                      "raw_T": evaluate.raw_layer_score(res, "T", msgs),
                      "raw_S": evaluate.raw_layer_score(res, "S", msgs),
                      "copypaste": baselines.copypaste(msgs), "luceri": extra["luceri"],
                      "account_machine": baselines.account_machine(msgs[msgs.kind != "post"], prob)}
            pc = pd.concat([evaluate.per_campaign_recall(scores, lab, a, group=gen_of) for a in (0.001, 0.01)])
            pc.insert(0, "level", level)
            pc.insert(1, "seed", seed)
            pc.to_csv(OUT / f"percampaign_{level}_s{seed}.csv", index=False)
            table.to_csv(f, index=False)
            print(level, seed, table.set_index("detector")[["recall@0.001", "benign_flag@0.01"]].round(3)
                  .to_dict(), flush=True)


if __name__ == "__main__":
    main()
