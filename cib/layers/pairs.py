"""Message-pair candidates and their aggregation into account-pair counts.

A layer produces message pairs (ia, ib) with an episode code; the account-pair statistic `obs` is the number of
distinct episodes in which the two accounts have at least one qualifying pair. The null model relabels accounts
and calls `account_counts` again on the same message pairs, so similarities are never recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MessagePairs:
    ia: np.ndarray       # int64 row positions into the message table
    ib: np.ndarray
    episode: np.ndarray  # int64 episode code

    def __len__(self) -> int:
        return len(self.ia)

    @staticmethod
    def empty() -> "MessagePairs":
        z = np.zeros(0, np.int64)
        return MessagePairs(z, z.copy(), z.copy())

    @staticmethod
    def concat(parts: list["MessagePairs"]) -> "MessagePairs":
        parts = [p for p in parts if len(p)]
        if not parts:
            return MessagePairs.empty()
        return MessagePairs(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("ia", "ib", "episode")))


def account_codes(accounts: pd.Series) -> tuple[np.ndarray, pd.Index]:
    codes, uniq = pd.factorize(accounts, sort=True)
    return codes.astype(np.int64), pd.Index(uniq)


def account_counts(pairs: MessagePairs, acc: np.ndarray, n_acc: int) -> pd.Series:
    """obs per account pair, indexed by key u * n_acc + v with u < v; same-account pairs are ignored."""
    if not len(pairs):
        return pd.Series(dtype=np.int64)
    a, b = acc[pairs.ia], acc[pairs.ib]
    keep = a != b
    u, v = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
    key = u * n_acc + v
    ep = pairs.episode[keep]
    # distinct (pair, episode) first, then count episodes per pair
    both = np.unique(np.stack([key, ep]), axis=1)
    k, c = np.unique(both[0], return_counts=True)
    return pd.Series(c, index=k)


def decode(keys: np.ndarray, n_acc: int) -> tuple[np.ndarray, np.ndarray]:
    return keys // n_acc, keys % n_acc


def windowed_pairs(ts: np.ndarray, window: int, group: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All index pairs (i<j in time order) with |ts_i - ts_j| <= window, optionally only inside equal `group`."""
    order = np.lexsort((ts,) if group is None else (ts, group))
    t = ts[order]
    g = np.zeros(len(t), np.int64) if group is None else group[order]
    ia, ib = [], []
    hi = np.searchsorted(t + (g * (10**12)), t + window + (g * (10**12)), side="right")
    for i in range(len(t)):
        if hi[i] > i + 1:
            j = np.arange(i + 1, hi[i])
            ia.append(np.full(len(j), i))
            ib.append(j)
    if not ia:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    return order[ia], order[ib]
