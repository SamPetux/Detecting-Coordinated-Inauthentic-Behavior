"""Build data/interim/vk_messages.parquet and an aggregate data report.

Confirmatory data are only counted (rows, accounts); distributions are computed on the exploratory split alone.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from cib import io_vk, schema


def explore_profile(df: pd.DataFrame) -> list[str]:
    acts = df.groupby("account_id").size()
    posts = df[df.kind == "post"].set_index("msg_id").ts
    com = df[df.kind != "post"]
    first = com.groupby("target_id").ts.min()
    delay = (first - posts.reindex(first.index)).dropna() / 60
    q = lambda s, p: f"{np.percentile(s, p):.0f}"
    hour = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(io_vk.TZ).dt.hour.value_counts(normalize=True)
    return [
        f"| Actions per account: median / p90 / p99 / max | {q(acts, 50)} / {q(acts, 90)} / {q(acts, 99)} / {acts.max()} |",
        f"| Accounts with 1 action / ≥10 / ≥100 | {(acts == 1).sum()} / {(acts >= 10).sum()} / {(acts >= 100).sum()} |",
        f"| Share of replies among comments | {(com.kind == 'reply').mean():.3f} |",
        f"| Community-authored share of actions | {df.author_is_community.mean():.3f} |",
        f"| Delay post → first comment, min: median / p90 | {q(delay, 50)} / {q(delay, 90)} |",
        f"| Comments per post: median / p90 | {q(com.groupby('target_id').size(), 50)} / {q(com.groupby('target_id').size(), 90)} |",
        f"| Busiest hour (MSK) share / quietest hour share | {hour.max():.3f} / {hour.min():.3f} |",
        f"| Texts shorter than 20 chars (comments) | {(com.text.str.len() < 20).mean():.3f} |",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/raw/corpus_cib_2026-11.csv.gz")
    ap.add_argument("--config", default="configs/explore.yaml")
    ap.add_argument("--out", default="data/interim/vk_messages.parquet")
    ap.add_argument("--report", default="docs/data_report.md")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["splits"]
    df = io_vk.load(args.corpus, cfg["explore_end"], cfg["confirm_end"])
    schema.validate(df)
    df.to_parquet(args.out, index=False)

    lines = ["# VK data report (aggregates only)", "",
             f"Source: `{args.corpus}` → `{args.out}`. Split boundaries (MSK): explore < {cfg['explore_end']}, "
             f"confirm < {cfg['confirm_end']}; later messages dropped.", "",
             "| Split | Posts | Comments | Replies | Accounts | Communities |", "|---|---|---|---|---|---|"]
    for sp, g in df.groupby("split"):
        k = g.kind.value_counts()
        lines.append(f"| {sp} | {k.get('post', 0)} | {k.get('comment', 0)} | {k.get('reply', 0)} | "
                     f"{g.account_id.nunique()} | {g.community.nunique()} |")
    ex = df[df.split == "explore"]
    lines += ["", "## Exploratory split profile", "", "| Statistic | Value |", "|---|---|", *explore_profile(ex), "",
              "Confirmatory split: counts only until the protocol is frozen (T16)."]
    open(args.report, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
