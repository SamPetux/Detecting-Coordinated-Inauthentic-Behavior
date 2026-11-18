"""T07 on the full exploratory set: can injected accounts be told apart by activity profile alone?"""
from __future__ import annotations

import pandas as pd

from cib.generate import inject
from scripts.explore_runs import load_synthetic, subset


def main() -> None:
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[bg.split == "explore"]
    syn = load_synthetic()
    rows = []
    for level in ("L0", "L1", "L2", "L3", "L4", "L5"):
        sub = subset(syn, level, "A", 1)
        sub = sub[sub.campaign_id.notna()]
        r = inject.artifact_audit(inject.inject(bg, sub))
        rows.append({"level": level, **r})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("docs/results/artifact_audit_explore.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
