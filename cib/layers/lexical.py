"""Layer L: near-duplicate texts (Jaccard of character shingles >= min_jaccard) within `window_s`.

Exact Jaccard on sparse shingle sets per time block; blocks are small enough that MinHash is unnecessary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from cib.layers.pairs import MessagePairs
from cib.text import clean_for_similarity


def message_pairs(msgs: pd.DataFrame, window_s: int = 3600, min_jaccard: float = 0.6, shingle: int = 5,
                  min_chars: int = 20) -> MessagePairs:
    text = msgs.text.fillna("").map(lambda t: clean_for_similarity(t).lower())
    has = (text.str.len() >= min_chars).to_numpy() & msgs.kind.ne("post").to_numpy()
    pos = np.flatnonzero(has)
    ts = msgs.ts.to_numpy()[pos].astype(np.int64)
    order = np.argsort(ts, kind="stable")
    pos, ts = pos[order], ts[order]
    if not len(pos):
        return MessagePairs.empty()
    vec = CountVectorizer(analyzer="char", ngram_range=(shingle, shingle), binary=True, dtype=np.float32)
    X = vec.fit_transform(text.to_numpy()[pos]).tocsr()
    size = np.asarray(X.sum(axis=1)).ravel()
    block = (ts - ts.min()) // window_s
    starts = np.searchsorted(block, np.arange(block.max() + 2))
    out = []
    for b in range(len(starts) - 2):
        s0, s1, s2 = starts[b], starts[b + 1], starts[b + 2]
        if s1 == s0:
            continue
        inter = (X[s0:s1] @ X[s0:s2].T).tocoo()
        i, j = inter.row + s0, inter.col + s0
        jac = inter.data / (size[i] + size[j] - inter.data)
        keep = (j > i) & (jac >= min_jaccard) & (ts[j] - ts[i] <= window_s)
        i, j = i[keep], j[keep]
        out.append(MessagePairs(pos[i], pos[j], (ts[i] // 3600).astype(np.int64)))
    return MessagePairs.concat(out)
