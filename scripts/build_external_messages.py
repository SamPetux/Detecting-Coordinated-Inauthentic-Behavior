"""Build BotSim-24 and fox8-23 message tables plus an aggregate report."""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

from cib import io_botsim, io_fox8, schema


def summary(name: str, df: pd.DataFrame) -> list[str]:
    lab = schema.account_labels(df)
    out = []
    for sp, g in df.groupby("split"):
        pos = g.campaign_id.notna()
        k = g.kind.value_counts().to_dict()
        t = pd.to_datetime(g.ts, unit="s", utc=True)
        out.append(f"| {name} | {sp} | {g.loc[pos, 'account_id'].nunique()} | {g.loc[~pos, 'account_id'].nunique()} | "
                   f"{pos.sum()} | {(~pos).sum()} | {k} | {t.min():%Y-%m-%d} – {t.max():%Y-%m-%d} |")
    out.append(f"| {name} | all | {int(lab.sum())} | {int((~lab).sum())} | | | | |")
    return out


def main() -> None:
    load_dotenv()
    salt = os.environ["CIB_SALT"]
    bs = io_botsim.load("data/raw/botsim/user_post_comment.json", "data/raw/botsim/Users.csv", salt)
    fx = io_fox8.load("data/raw/fox8/fox8_23_dataset.ndjson.gz", salt)
    for name, df in (("botsim", bs), ("fox8", fx)):
        schema.validate(df)
        df.to_parquet(f"data/interim/{name}_messages.parquet", index=False)
    lines = ["# External data report (aggregates only)", "",
             "| Dataset | Split | Bot accounts | Human accounts | Bot actions | Human actions | Kinds | Period |",
             "|---|---|---|---|---|---|---|---|", *summary("botsim", bs), *summary("fox8", fx), "",
             "fox8 humans and bots are not co-temporal: no false-positive rates on fox8."]
    open("docs/data_report_external.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
