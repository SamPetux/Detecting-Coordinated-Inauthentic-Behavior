"""Message table: one row per action, shared by every data source, generator and layer."""
from __future__ import annotations

import hashlib

import pandas as pd

KINDS = frozenset({"post", "comment", "reply", "repost"})
SPLITS = frozenset({"explore", "confirm"})

# column -> pandas dtype; list columns are kept as object
COLUMNS: dict[str, str] = {
    "msg_id": "string",
    "account_id": "string",
    "ts": "int64",
    "kind": "string",
    "target_id": "string",
    "parent_id": "string",
    "community": "string",
    "text": "string",
    "hashtags": "object",
    "urls": "object",
    "source": "string",
    "is_injected": "bool",
    "author_is_community": "bool",
    "campaign_id": "string",
    "level": "string",
    "split": "string",
}
REQUIRED = ("msg_id", "account_id", "ts", "kind", "target_id", "community", "source", "split")
ACCOUNT_ID_LEN = 16


def hash_account(raw: object, salt: str) -> str:
    """Pseudonymise a platform account id; synthetic ids go through the same function to look identical."""
    return hashlib.sha256(f"{salt}{raw}".encode()).hexdigest()[:ACCOUNT_ID_LEN]


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNS.items()})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns, cast dtypes and order columns."""
    out = df.copy()
    for c, t in COLUMNS.items():
        if c not in out:
            out[c] = None if t == "object" else pd.Series(pd.NA, index=out.index, dtype=t if t != "bool" else "boolean")
    for c in ("is_injected", "author_is_community"):
        out[c] = out[c].fillna(False).astype(bool)
    for c, t in COLUMNS.items():
        if t not in ("object", "bool"):
            out[c] = out[c].astype(t)
    return out[list(COLUMNS)]


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError on any contract violation."""
    missing = [c for c in COLUMNS if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    nulls = [c for c in REQUIRED if df[c].isna().any()]
    if nulls:
        raise ValueError(f"nulls in required columns: {nulls}")
    if df.msg_id.duplicated().any():
        raise ValueError("duplicate msg_id")
    bad_kind = set(df.kind.unique()) - KINDS
    if bad_kind:
        raise ValueError(f"unknown kind: {bad_kind}")
    bad_split = set(df.split.unique()) - SPLITS
    if bad_split:
        raise ValueError(f"unknown split: {bad_split}")
    if (df.account_id.str.len() != ACCOUNT_ID_LEN).any():
        raise ValueError("account_id must be a 16-char hash")
    inj = df.is_injected
    if df.loc[inj, "campaign_id"].isna().any() and df.loc[inj, "level"].isna().any():
        raise ValueError("injected rows need campaign_id or level")


def account_labels(df: pd.DataFrame) -> pd.Series:
    """True for accounts with at least one campaign action; benign-LLM controls have no campaign_id."""
    return df.groupby("account_id")["campaign_id"].agg(lambda s: s.notna().any())
