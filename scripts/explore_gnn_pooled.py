"""Exploration: one graph-only GNN trained on seed-1 runs of all levels at once, tested on seed 2 of each level."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import yaml

from cib import evaluate, gnn, metrics
from scripts.explore_runs import load_embeddings, load_synthetic, set_path, subset

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")


def main() -> None:
    cfg = yaml.safe_load(open("configs/explore.yaml"))
    for kv in ("null_model.bucket=community_day", "fusion.account_agg=top3"):
        set_path(cfg, *kv.split("=", 1))
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[bg.split == "explore"].reset_index(drop=True)
    syn = load_synthetic()
    emb, row_of = load_embeddings(bg, syn)
    prob = pd.read_parquet("data/interim/machine_prob_explore.parquet").p
    runs = {}
    for seed in (1, 2):
        for level in LEVELS:
            _, res, lab = evaluate.run_once(bg, subset(syn, level, "A", seed), copy.deepcopy(cfg), emb=emb,
                                            row_of=row_of, machine_prob=prob, log=lambda s: None)
            runs[(level, seed)] = (res, lab)
            print("run", level, seed, flush=True)
    cols = sorted({c for r, _ in runs.values() for c in r.fused.columns if c.startswith("nl_")})
    graphs, ys = [], []
    for level in LEVELS:
        res, lab = runs[(level, 1)]
        d, names = gnn.build_graph(res.fused, cols, pd.DataFrame(index=lab.index))
        graphs.append(d)
        ys.append(lab.positive.reindex(names).fillna(False).to_numpy().astype(float))
    big, y = gnn.union(graphs, ys)
    rng = np.random.default_rng(0)
    tr = rng.random(len(y)) < 0.8
    g = cfg["gnn"]
    model = gnn.train(big, y, tr, val_mask=~tr, seed=0, hidden=g["hidden"], layers=g["layers"], dropout=g["dropout"],
                      conv=g["conv"])
    rows = []
    for level in LEVELS:
        res, lab = runs[(level, 2)]
        d, names = gnn.build_graph(res.fused, cols, pd.DataFrame(index=lab.index))
        p = gnn.predict(model, d, names)
        t = evaluate.score_table({"gnn_pooled": p}, lab)
        t.insert(0, "level", level)
        rows.append(t)
        print(t[["level", "recall@0.001", "recall@0.01", "benign_flag@0.001", "benign_flag@0.01"]].round(3)
              .to_string(index=False, header=level == "L0"), flush=True)
    pd.concat(rows).to_csv("docs/results/explore_gnn_pooled.csv", index=False)


if __name__ == "__main__":
    main()
