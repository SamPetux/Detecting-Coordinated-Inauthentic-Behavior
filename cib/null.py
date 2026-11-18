"""Bucket-preserving account permutation (BPAP) null model and per-edge significance.

Within each bucket (community x time slot) account labels are shuffled among the non-post messages. Preserved:
bucket volumes (organic bursts), each account's number of messages per bucket (its rhythm), texts and targets.
Broken: which of the accounts present wrote the similar text / commented the same post at the same moment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from cib.layers.pairs import MessagePairs, account_counts, decode


def buckets(msgs: pd.DataFrame, slot_s: int = 3600, by_community: bool = True) -> np.ndarray:
    slot = (msgs.ts.to_numpy() // slot_s).astype(np.int64)
    if not by_community:
        return slot
    com, _ = pd.factorize(msgs.community.to_numpy())
    return com.astype(np.int64) * (10**7) + (slot - slot.min())


def permute_within(acc: np.ndarray, bucket: np.ndarray, movable: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = acc.copy()
    idx = np.flatnonzero(movable)
    b = bucket[idx]
    by_bucket = idx[np.argsort(b, kind="stable")]
    shuffled = idx[np.lexsort((rng.random(len(idx)), b))]
    out[by_bucket] = acc[shuffled]
    return out


def benjamini_hochberg(p: np.ndarray, m: int | None = None) -> np.ndarray:
    """BH q-values; `m` = total number of hypotheses when only a pre-selected subset of p-values is passed
    (e.g. kNN edges chosen out of all account pairs)."""
    n = len(p)
    if not n:
        return p
    m = n if m is None else max(m, n)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def edge_significance(pairs: MessagePairs, acc: np.ndarray, n_acc: int, bucket: np.ndarray, movable: np.ndarray,
                      *, permutations: int = 50, min_obs: int = 2, fdr_q: float = 0.01, seed: int = 0,
                      global_null: bool = False) -> pd.DataFrame:
    """Edges with obs >= min_obs: u, v, obs, null_mean, p (Poisson tail), q (BH within layer), sig.

    global_null=True permutes across the whole table (does not preserve bursts); used as the H3 contrast.
    """
    obs = account_counts(pairs, acc, n_acc)
    obs = obs[obs >= min_obs]
    if obs.empty:
        return pd.DataFrame(columns=["u", "v", "obs", "null_mean", "p", "q", "sig"])
    rng = np.random.default_rng(seed)
    b = np.zeros_like(bucket) if global_null else bucket
    total = np.zeros(len(obs))
    for _ in range(permutations):
        perm = permute_within(acc, b, movable, rng)
        total += account_counts(pairs, perm, n_acc).reindex(obs.index, fill_value=0).to_numpy()
    null_mean = total / permutations
    # +1/R keeps the tail finite when an edge never appeared in any permutation
    p = poisson.sf(obs.to_numpy() - 1, null_mean + 1.0 / permutations)
    q = benjamini_hochberg(p)
    u, v = decode(obs.index.to_numpy(), n_acc)
    return pd.DataFrame({"u": u, "v": v, "obs": obs.to_numpy(), "null_mean": null_mean, "p": p, "q": q,
                         "sig": q <= fdr_q})
