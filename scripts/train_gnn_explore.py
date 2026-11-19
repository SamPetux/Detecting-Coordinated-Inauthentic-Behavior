"""Protocol M*: one graph-only GNN trained on all exploratory runs (levels L0-L5, seeds 1-2), saved for confirmation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cib import evaluate, gnn
from scripts.explore_runs import load_embeddings, load_synthetic, subset

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
OUT = Path("results/models")


def main() -> None:
    cfg = yaml.safe_load(open("configs/confirm.yaml"))
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[bg.split == "explore"].reset_index(drop=True)
    syn = load_synthetic()
    emb, row_of = load_embeddings(bg, syn)
    prob = pd.read_parquet("data/interim/machine_prob_explore.parquet").p
    graphs, ys, cols = [], [], None
    runs = []
    for seed in (1, 2):
        for level in LEVELS:
            _, res, lab = evaluate.run_once(bg, subset(syn, level, "A", seed), copy.deepcopy(cfg), emb=emb,
                                            row_of=row_of, machine_prob=prob, log=lambda s: None)
            runs.append((res, lab))
            print("run", level, seed, flush=True)
    cols = sorted({c for r, _ in runs for c in r.fused.columns if c.startswith("nl_")})
    for res, lab in runs:
        d, names = gnn.build_graph(res.fused, cols, pd.DataFrame(index=lab.index))
        graphs.append(d)
        ys.append(lab.positive.reindex(names).fillna(False).to_numpy().astype(float))
    big, y = gnn.union(graphs, ys)
    rng = np.random.default_rng(0)
    tr = rng.random(len(y)) < 0.8
    g = cfg["gnn"]
    model = gnn.train(big, y, tr, val_mask=~tr, seed=0, hidden=g["hidden"], layers=g["layers"], dropout=g["dropout"],
                      conv=g["conv"])
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "gnn_pooled.pt")
    (OUT / "gnn_pooled.json").write_text(json.dumps({"edge_cols": cols, "in_dim": 0, "edge_dim": len(cols), **g}))
    print("saved", OUT / "gnn_pooled.pt", "edge cols", cols, flush=True)


if __name__ == "__main__":
    main()
