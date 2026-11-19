"""Baselines. Each returns an account score (pd.Series indexed by account_id); thresholds are set later at an equal
organic false-alarm rate (cib.metrics.recall_at_fpr), as in the main method.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from cib.layers import cotarget, lexical
from cib.layers.pairs import MessagePairs, account_codes, account_counts, decode

LUCERI_MODEL = "sentence-transformers/stsb-xlm-r-multilingual"
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
                    "✂-➰Ⓜ-\U0001F251]+")


def degree(counts: pd.Series, names: pd.Index, n: int) -> pd.Series:
    """Weighted degree from account-pair counts keyed by u*n+v."""
    if counts.empty:
        return pd.Series(dtype=float)
    u, v = decode(counts.index.to_numpy(), n)
    w = counts.to_numpy(dtype=float)
    deg = np.bincount(u, w, minlength=n) + np.bincount(v, w, minlength=n)
    return pd.Series(deg, index=names)


def luceri_clean(text: str) -> str:
    """Their msg_clean (URLs, mentions, digits, emoji, punctuation) minus the English stop-word list."""
    t = _URL.sub(" ", text or "")
    t = re.sub(r"@\w+|\[(?:id|club|public)\d+\|[^\]]*\]", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"[:;.,!&\-_$/?]", "", t).lower()
    return " ".join(t.split())


def luceri_text_similarity(msgs: pd.DataFrame, emb: np.ndarray, row_of: pd.Series, *, threshold: float = 0.7,
                           window_days: int = 1) -> pd.Series:
    """Luceri et al. 2024 text-similarity network: cosine >= threshold between texts (>4 words after cleaning)
    in day-long windows; edge weight = number of similar pairs. Embeddings must come from LUCERI_MODEL over
    `luceri_clean` texts. Score = weighted degree."""
    sel = np.flatnonzero(msgs.msg_id.isin(row_of.index).to_numpy() & msgs.kind.ne("post").to_numpy())
    m = msgs.iloc[sel]
    day = (m.ts.to_numpy() // 86400 // window_days).astype(np.int64)
    acc, names = account_codes(msgs.account_id)
    accm = acc[sel]
    vec = emb[row_of.loc[m.msg_id].to_numpy()]
    ia, ib = [], []
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        sim = vec[idx] @ vec[idx].T
        i, j = np.nonzero(np.triu(sim >= threshold, 1))
        ia.append(idx[i])
        ib.append(idx[j])
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    # every similar pair counts (no episode collapsing), as in their value_counts over pairs
    pairs = MessagePairs(ia, ib, np.arange(len(ia), dtype=np.int64))
    return degree(account_counts(pairs, accm, len(names)), names, len(names))


def raw_cotarget(msgs: pd.DataFrame, window_s: int = 600) -> pd.Series:
    """Coordination window on the same post without a null model (CooRnet-style)."""
    acc, names = account_codes(msgs.account_id)
    return degree(account_counts(cotarget.message_pairs(msgs, window_s), acc, len(names)), names, len(names))


def raw_pairs(msgs: pd.DataFrame, pairs: MessagePairs) -> pd.Series:
    acc, names = account_codes(msgs.account_id)
    return degree(account_counts(pairs, acc, len(names)), names, len(names))


def copypaste(msgs: pd.DataFrame, window_s: int = 3600, min_jaccard: float = 0.9) -> pd.Series:
    return raw_pairs(msgs, lexical.message_pairs(msgs, window_s=window_s, min_jaccard=min_jaccard))


def account_machine(msgs: pd.DataFrame, prob: pd.Series) -> pd.Series:
    """Individual-level baseline: mean machine-generation probability of an account's messages."""
    p = prob.reindex(msgs.msg_id).to_numpy()
    return pd.Series(p, index=msgs.account_id.to_numpy()).dropna().groupby(level=0).mean()
