"""Aggregate exploration runs (results/explore/<tag>/*.csv) into docs/results/explore_<tag>.md (aggregates only)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLS = ["recall@0.001", "recall@0.01", "auprc", "benign_flag@0.01", "h3_ratio@0.01"]
ORDER = ["full", "gnn_all", "gnn_style", "gnn_none", "full_noM", "full_noMY", "layer_T", "layer_S", "layer_L", "raw_T", "raw_S", "copypaste",
         "account_machine", "luceri"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    files = sorted(Path(f"results/explore/{args.tag}").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["run"] = df.generator + "-s" + df.seed.astype(str)
    lines = [f"# Exploration summary: `{args.tag}`", "",
             f"Runs: {len(files)} files. Mean over seeds (generator A); generator C (unseen) shown separately.", ""]
    for gen in ("A", "C"):
        sub = df[df.generator == gen]
        if sub.empty:
            continue
        cols = [c for c in COLS if c in sub]
        piv = sub.groupby(["detector", "level"])[cols].mean()
        for col in cols:
            t = piv[col].unstack("level")
            t = t.reindex([d for d in ORDER if d in t.index] + [d for d in t.index if d not in ORDER])
            lines += [f"## Generator {gen}: {col}", "", t.round(3).to_markdown(), ""]
    out = Path(f"docs/results/explore_{args.tag}.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
