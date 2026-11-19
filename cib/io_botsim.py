"""BotSim-24 -> message table.

The released JSON keeps first- and second-level comments under `comment_1` / `comment_2` (not `comments` as the
dataset README says). Users.csv lists 1,907 humans first and 1,000 LLM bots last. Humans and bots share one
period (2023-06-20 .. 2024-06-19), so the data are split by time, like the VK corpus.
"""
from __future__ import annotations

import json

import pandas as pd

from cib import schema

N_HUMANS = 1907
CAMPAIGN = "botsim"


def _ts(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True)
    return (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def from_raw(data: dict, users: pd.DataFrame, salt: str, split_at: str) -> pd.DataFrame:
    bots = set(users.user_id.iloc[N_HUMANS:])
    rows = []
    for uid, acts in data.items():
        acc = schema.hash_account(uid, salt)
        camp = CAMPAIGN if uid in bots else None
        for p in acts.get("posts", []):
            rows.append((f"t3_{p['submission_id']}", acc, p["created_utc"], "post", p["submission_id"], None,
                         p["subreddit"], p.get("posts") or "", camp))
        for key in ("comment_1", "comment_2"):
            for c in acts.get(key, []):
                parent = c.get("parent_id") or ""
                is_reply = parent.startswith("t1_")
                rows.append((f"t1_{c['comment_id']}", acc, c["created_utc"], "reply" if is_reply else "comment",
                             c["link_id"].removeprefix("t3_"), parent if is_reply else None, c["subreddit"],
                             c.get("comment_body") or "", camp))
    df = pd.DataFrame(rows, columns=["msg_id", "account_id", "created", "kind", "target_id", "parent_id",
                                     "community", "text", "campaign_id"])
    df["ts"] = _ts(df.pop("created"))
    df["source"] = "botsim"
    df["split"] = (df.ts < _ts(pd.Series([split_at])).iloc[0]).map({True: "explore", False: "confirm"})
    df = df.drop_duplicates("msg_id")
    return schema.conform(df.sort_values(["ts", "msg_id"], kind="stable").reset_index(drop=True))


def load(json_path: str, users_path: str, salt: str, split_at: str = "2023-12-20T00:00:00Z") -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    return from_raw(data, pd.read_csv(users_path), salt, split_at)
