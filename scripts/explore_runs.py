"""T15 driver: evaluation runs on the exploratory split for one config (resumable; one CSV per run).

    python -m scripts.explore_runs --config configs/explore.yaml --tag base
    python -m scripts.explore_runs --set layers.semantic.min_cos=0.94 --tag tau094 --levels L3 L4
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cib import evaluate

RUNS = [(lvl, "A", 1) for lvl in ("L0", "L1", "L2", "L3", "L4", "L5")] + \
       [(lvl, "A", 2) for lvl in ("L0", "L1", "L2", "L3", "L4", "L5")] + [("L3", "C", 3)]
BENIGN = {"A": "benign-explore-s11", "C": "benign-explore-s12"}


def set_path(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = yaml.safe_load(value)


def load_embeddings(bg: pd.DataFrame, syn: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    from cib import embed
    xb, ib = embed.embed_messages(bg, "intfloat/multilingual-e5-small", cache_dir="data/interim/emb/vk_explore")
    xs, is_ = embed.embed_messages(syn, "intfloat/multilingual-e5-small", cache_dir="data/interim/emb/syn_explore")
    x = np.vstack([xb, xs])
    row_of = pd.Series(np.arange(len(x)), index=ib.append(is_))
    eligible = pd.concat([bg, syn]).set_index("msg_id").text.map(lambda t: len(embed.clean_for_similarity(t)) >= 20)
    return x, row_of[row_of.index.isin(eligible.index[eligible])]


def subset(syn: pd.DataFrame, level: str, gen: str, seed: int) -> pd.DataFrame:
    cid = syn.campaign_id.fillna("")
    camp = cid.str.contains(f"-{level}-s{seed}") & cid.str.endswith(f"-{gen}")
    return syn[camp | syn.msg_id.str.contains(BENIGN[gen])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/explore.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--no-machine", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_path(cfg, k, v)
    out = Path(f"results/explore/{args.tag}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))

    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[bg.split == "explore"].reset_index(drop=True)
    syn = pd.read_parquet("data/synthetic/explore/messages.parquet")
    emb, row_of = load_embeddings(bg, syn)
    prob = None
    if not args.no_machine:
        prob = pd.read_parquet("data/interim/machine_prob_explore.parquet").p
    keep = {}
    for level, gen, seed in RUNS:
        if args.levels and level not in args.levels:
            continue
        f = out / f"{level}_{gen}_s{seed}.csv"
        if f.exists():
            continue
        table, res, lab = evaluate.run_once(bg, subset(syn, level, gen, seed), copy.deepcopy(cfg), emb=emb,
                                            row_of=row_of, machine_prob=prob, log=lambda s: print(f"  {s}", flush=True))
        table.insert(0, "level", level)
        table.insert(1, "generator", gen)
        table.insert(2, "seed", seed)
        table.to_csv(f, index=False)
        keep[(level, gen, seed)] = (res, lab)
        print(f"{level} {gen} s{seed}:", table.set_index("detector")[["recall@0.001", "recall@0.01",
                                                                            "benign_flag@0.001"]].round(3).to_dict(),
              flush=True)


if __name__ == "__main__":
    main()
