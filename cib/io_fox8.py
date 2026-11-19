"""fox8-23 -> message table.

Humans come from 2009-2020 bot-detection datasets while the bots run until 2023, so the two classes are not
co-temporal. Use only for analysis inside the botnet and for the machine-generation layer; never for
false-positive rates. Everything is marked as `confirm` because nothing here may be used for tuning.
"""
from __future__ import annotations

import gzip
import json

import pandas as pd

from cib import schema

CAMPAIGN = "fox8"


def _row(tw: dict, acc: str, camp: str | None) -> tuple:
    rt = tw.get("retweeted_status")
    if rt:
        kind, target, parent = "repost", rt["id_str"], None
    elif tw.get("in_reply_to_status_id_str"):
        kind, target, parent = "reply", tw["in_reply_to_status_id_str"], tw["in_reply_to_status_id_str"]
    else:
        kind, target, parent = "post", tw["id_str"], None
    ent = tw.get("entities") or {}
    return (tw["id_str"], acc, tw["created_at"], kind, target, parent, "" if rt else (tw.get("full_text") or tw.get("text") or ""),
            [h["text"] for h in ent.get("hashtags", [])], [u.get("expanded_url") for u in ent.get("urls", [])], camp)


def load(path: str, salt: str) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            o = json.loads(line)
            acc = schema.hash_account(o["user_id"], salt)
            camp = CAMPAIGN if o["label"] == "bot" else None
            rows.extend(_row(tw, acc, camp) for tw in o["user_tweets"])
    df = pd.DataFrame(rows, columns=["msg_id", "account_id", "created", "kind", "target_id", "parent_id", "text",
                                     "hashtags", "urls", "campaign_id"])
    t = pd.to_datetime(df.pop("created"), format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df["ts"] = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    df["community"] = "twitter"
    df["source"] = "fox8"
    df["split"] = "confirm"
    df = df.drop_duplicates("msg_id")
    return schema.conform(df.sort_values(["ts", "msg_id"], kind="stable").reset_index(drop=True))
