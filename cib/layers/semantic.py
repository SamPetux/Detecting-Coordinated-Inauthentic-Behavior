"""Layer S: embedding cosine >= min_cos within `window_s`, any target; episode = hour of the earlier message.

Blocks of `window_s` are compared with themselves and the next block, so every pair with |dt| <= window_s is seen
once. Blocks are small (tens to hundreds of messages per hour), so a dense product is cheaper than an ANN index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cib.layers.pairs import MessagePairs


def message_pairs(msgs: pd.DataFrame, emb: np.ndarray, row_of: pd.Series, window_s: int = 3600,
                  min_cos: float = 0.85) -> MessagePairs:
    """`emb[row_of[msg_id]]` is the normalised vector of a message; messages without a row are skipped."""
    has = msgs.msg_id.isin(row_of.index).to_numpy() & msgs.kind.ne("post").to_numpy()
    pos = np.flatnonzero(has)
    ts = msgs.ts.to_numpy()[pos].astype(np.int64)
    order = np.argsort(ts, kind="stable")
    pos, ts = pos[order], ts[order]
    vec = emb[row_of.loc[msgs.msg_id.to_numpy()[pos]].to_numpy()]
    block = (ts - ts.min()) // window_s if len(ts) else ts
    starts = np.searchsorted(block, np.arange(block.max() + 2 if len(block) else 0))
    out = []
    for b in range(len(starts) - 2):
        s0, s1, s2 = starts[b], starts[b + 1], starts[b + 2]
        if s1 == s0:
            continue
        sim = vec[s0:s1] @ vec[s0:s2].T
        i, j = np.nonzero(sim >= min_cos)
        j = j + s0
        i = i + s0
        keep = (j > i) & (ts[j] - ts[i] <= window_s)
        i, j = i[keep], j[keep]
        out.append(MessagePairs(pos[i], pos[j], (ts[i] // 3600).astype(np.int64)))
    return MessagePairs.concat(out)
