"""Cluster bootstrap, paired Wilcoxon tests and Holm correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cluster_bootstrap(values: pd.Series, clusters: pd.Series, stat=np.mean, n: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimate, lo, hi): resample whole clusters (campaigns/series) with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g.to_numpy() for _, g in values.groupby(clusters)]
    est = float(stat(values.to_numpy()))
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        boot[i] = stat(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    """(median difference a-b, p). Zero differences are kept (zsplit) so ties do not inflate significance."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 0.0, 1.0
    return float(np.median(d)), float(wilcoxon(a, b, zero_method="zsplit", alternative=alternative).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
