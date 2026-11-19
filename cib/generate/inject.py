"""Injection of synthetic campaigns into a background split and the trivial-artifact audit.

The audit asks whether injected accounts can be told from organic ones by features unrelated to synchronisation.
If they can, any detector (ours or a baseline) may be rewarded for an artefact of the simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from cib import schema


def inject(background: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([background, synthetic], ignore_index=True)
    df = schema.conform(df.sort_values(["ts", "msg_id"], kind="stable").reset_index(drop=True))
    schema.validate(df)
    return df


def trivial_features(msgs: pd.DataFrame) -> pd.DataFrame:
    """Per-account features that a synchronisation detector should not need."""
    m = msgs[msgs.kind != "post"]
    hour = (m.ts // 3600) % 24
    g = m.groupby("account_id")
    feats = pd.DataFrame({
        "n_actions": g.size(),
        "span_h": (g.ts.max() - g.ts.min()) / 3600,
        "reply_share": g.kind.apply(lambda k: (k == "reply").mean()),
        "n_targets": g.target_id.nunique(),
        "n_communities": g.community.nunique(),
        "sec_of_min_std": g.ts.apply(lambda t: float(np.std(t % 60))),
        "hour_entropy": hour.groupby(m.account_id).apply(
            lambda h: float(-(h.value_counts(normalize=True) * np.log(h.value_counts(normalize=True))).sum())),
        "first_seen_rel": (g.ts.min() - m.ts.min()) / max(int(m.ts.max() - m.ts.min()), 1),
    })
    return feats


def artifact_audit(msgs: pd.DataFrame, seed: int = 0, with_length: bool = False) -> dict:
    """AUC of a 5-fold GBM separating injected campaign accounts from organic accounts matched on n_actions."""
    feats = trivial_features(msgs)
    if with_length:
        m = msgs[msgs.kind != "post"]
        feats["mean_len"] = m.text.str.len().groupby(m.account_id).mean()
    lab = msgs.groupby("account_id").is_injected.any().reindex(feats.index)
    inj, org = feats[lab], feats[~lab]
    # match organic accounts on activity so the audit does not just rediscover the long tail
    rng = np.random.default_rng(seed)
    pick = []
    for n, cnt in inj.n_actions.value_counts().items():
        pool = org.index[org.n_actions == n]
        if len(pool):
            pick.extend(rng.choice(pool, size=min(len(pool), 3 * cnt), replace=False))
    X = pd.concat([inj, org.loc[pick]])
    y = np.r_[np.ones(len(inj)), np.zeros(len(pick))]
    if len(pick) < 10 or len(inj) < 10:
        return {"auc": np.nan, "n_injected": len(inj), "n_matched": len(pick)}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X.fillna(0), y, cv=cv,
                          method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(y, p)), "n_injected": len(inj), "n_matched": len(pick)}
