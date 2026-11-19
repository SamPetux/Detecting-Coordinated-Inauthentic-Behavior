"""Protocol section 5 (descriptive): price of evasion at L4 — recall of M* and campaign visibility by median reaction
delay and participation share. Reads results/confirm/percampaign_L4_*.csv and the confirmatory synthetic data."""
from __future__ import annotations

import glob

import pandas as pd

from cib import metrics


def main() -> None:
    pc = pd.concat([pd.read_csv(f) for f in glob.glob("results/confirm/percampaign_L4_s*.csv")])
    pc = pc[(pc.fpr == 0.001) & pc.detector.isin(["M_star", "M0", "account_machine"])]
    pc["hit"] = pc.recall * pc.n
    rec = pc.groupby(["detector", "campaign"])[["hit", "n"]].sum()
    rec = (rec.hit / rec.n).rename("recall").reset_index()
    plan = pd.read_parquet("data/synthetic/confirm/plan.parquet")
    grid = plan.drop_duplicates("campaign_id").set_index("campaign_id")[["median_delay_s", "participation"]]
    rec = rec.join(grid, on="campaign")
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[bg.split == "confirm"]
    syn = pd.read_parquet("data/synthetic/confirm/messages.parquet")
    syn = syn[syn.campaign_id.fillna("").str.contains("-L4-") & ~syn.msg_id.str.contains("#hist")]
    vis = metrics.visibility(pd.concat([bg[bg.target_id.isin(syn.target_id)], syn])).rename("visibility")
    rec = rec.join(vis, on="campaign")
    rec["delay_min"] = (rec.median_delay_s / 60).round().astype(int)
    t = rec.pivot_table(index=["delay_min", "participation"], columns="detector", values=["recall", "visibility"],
                        aggfunc="mean").round(3)
    t.to_csv("docs/results/confirm_evasion.csv")
    with open("docs/results/confirm_evasion.md", "w") as f:
        f.write("# Price of evasion (L4, confirmatory, descriptive)\n\n" + t.to_markdown() + "\n")
    print(t.to_string())


if __name__ == "__main__":
    main()
