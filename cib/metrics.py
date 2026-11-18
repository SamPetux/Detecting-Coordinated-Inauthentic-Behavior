"""Evaluation metrics.

Recall@FPR: the score threshold is set on organic accounts only (accounts without campaign actions and not in the
benign-LLM control), so that the given share of organic accounts is flagged; recall is the share of campaign
accounts above it. Accounts never scored by a detector get score 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, normalized_mutual_info_score


def threshold_at_fpr(organic_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold with at most `fpr` share of organic scores strictly above it."""
    s = np.sort(organic_scores)[::-1]
    k = int(np.floor(fpr * len(s)))
    return s[k] if k < len(s) else -np.inf


def recall_at_fpr(score: pd.Series, positive: pd.Series, organic: pd.Series, fpr: float,
                  threshold: float | None = None) -> tuple[float, float]:
    """(recall, threshold). Pass `threshold` to reuse one fixed on calibration data."""
    s = score.reindex(positive.index, fill_value=0.0)
    thr = threshold_at_fpr(s[organic].to_numpy(), fpr) if threshold is None else threshold
    return float((s[positive] > thr).mean()), float(thr)


def flagged_rate(score: pd.Series, accounts: pd.Index, thr: float) -> float:
    return float((score.reindex(accounts, fill_value=0.0) > thr).mean())


def auprc(score: pd.Series, positive: pd.Series, negative: pd.Series) -> float:
    s = score.reindex(positive.index, fill_value=0.0)
    keep = positive | negative
    return float(average_precision_score(positive[keep], s[keep]))


def best_cluster_f1(clusters: pd.DataFrame, campaign_of: pd.Series) -> pd.Series:
    """F1* per campaign: best F1 over clusters (cluster_id >= 0) against that campaign's accounts."""
    cl = clusters[clusters.cluster_id >= 0].set_index("account_id").cluster_id
    out = {}
    for camp, accs in campaign_of.dropna().groupby(campaign_of.dropna()):
        members = set(accs.index)
        best = 0.0
        for cid, g in cl.groupby(cl):
            tp = len(members & set(g.index))
            if tp:
                p, r = tp / len(g), tp / len(members)
                best = max(best, 2 * p * r / (p + r))
        out[camp] = best
    return pd.Series(out, dtype=float)


def nmi(clusters: pd.DataFrame, campaign_of: pd.Series) -> float:
    """NMI between predicted clusters and campaigns over all accounts (unclustered/organic = one label each)."""
    pred = clusters.set_index("account_id").cluster_id.reindex(campaign_of.index).fillna(-1)
    pred = pred.where(pred >= 0, "none")
    truth = campaign_of.fillna("none")
    return float(normalized_mutual_info_score(truth.astype(str), pred.astype(str)))


def visibility(msgs: pd.DataFrame, top: int = 10) -> pd.Series:
    """Per campaign: share of its comments among the first `top` comments of the targeted posts."""
    c = msgs[msgs.kind != "post"].sort_values("ts")
    c = c.assign(rank=c.groupby("target_id").cumcount())
    camp = c[c.campaign_id.notna()]
    return camp.groupby("campaign_id")["rank"].apply(lambda r: float((r < top).mean()))
