"""Reaction-delay and participation models for campaign actions."""
from __future__ import annotations

import numpy as np


def delays(rng: np.random.Generator, n: int, median_s: float, sigma: float = 0.8) -> np.ndarray:
    """Log-normal reaction delays in seconds with a random seconds component (no round timestamps)."""
    d = rng.lognormal(mean=np.log(median_s), sigma=sigma, size=n)
    return np.maximum(d + rng.uniform(0, 60, size=n), 5).astype(np.int64)


def participation(rng: np.random.Generator, n_accounts: int, n_targets: int, share: float) -> np.ndarray:
    """Boolean (accounts x targets) mask; every target keeps at least two participants."""
    m = rng.random((n_accounts, n_targets)) < share
    for t in range(n_targets):
        if m[:, t].sum() < 2:
            m[rng.choice(n_accounts, size=2, replace=False), t] = True
    return m
