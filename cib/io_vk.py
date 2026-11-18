"""VK corpus export (scripts/export_vk_corpus.py) -> message table with explore/confirm splits."""
from __future__ import annotations

import pandas as pd

from cib import schema

TZ = "Europe/Moscow"
USECOLS = ["kind", "msg_key", "post_key", "parent_key", "group", "date_utc", "account", "author_is_community", "text"]


def split_of(ts_utc: pd.Series, explore_end: str, confirm_end: str) -> pd.Series:
    """'explore' before explore_end, 'confirm' before confirm_end, NA afterwards (dropped)."""
    e, c = pd.Timestamp(explore_end), pd.Timestamp(confirm_end)
    out = pd.Series(pd.NA, index=ts_utc.index, dtype="string")
    out[ts_utc < e] = "explore"
    out[(ts_utc >= e) & (ts_utc < c)] = "confirm"
    return out


def from_export(raw: pd.DataFrame, explore_end: str, confirm_end: str) -> pd.DataFrame:
    ts = pd.to_datetime(raw.date_utc, utc=True, format="ISO8601")
    is_post = raw.kind.eq("post")
    kind = pd.Series("comment", index=raw.index, dtype="string")
    kind[is_post] = "post"
    kind[~is_post & raw.parent_key.notna()] = "reply"
    df = pd.DataFrame({
        "msg_id": raw.msg_key,
        "account_id": raw.account,
        # pandas 3 parses to microseconds, so never assume ns when converting to epoch seconds
        "ts": (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1),
        "kind": kind,
        "target_id": raw.post_key,
        "parent_id": raw.parent_key,
        "community": raw.group,
        "text": raw.text.fillna(""),
        "source": "vk",
        "is_injected": False,
        "author_is_community": raw.author_is_community.astype(str).str.lower().eq("true"),
        "split": split_of(ts, explore_end, confirm_end),
    })
    df = df[df.split.notna() & df.account_id.notna()]
    return schema.conform(df.sort_values(["ts", "msg_id"], kind="stable").reset_index(drop=True))


def load(path: str, explore_end: str, confirm_end: str) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=USECOLS, dtype={"parent_key": "string", "account": "string"}, low_memory=False)
    return from_export(raw, explore_end, confirm_end)
