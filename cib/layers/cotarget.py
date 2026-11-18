"""Layer T: two accounts act on the same target within `window_s` seconds; episode = target."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cib.layers.pairs import MessagePairs, windowed_pairs


def message_pairs(msgs: pd.DataFrame, window_s: int = 600) -> MessagePairs:
    acts = msgs.kind.ne("post").to_numpy()
    pos = np.flatnonzero(acts)
    tgt, _ = pd.factorize(msgs.target_id.to_numpy()[pos])
    ia, ib = windowed_pairs(msgs.ts.to_numpy()[pos].astype(np.int64), window_s, tgt.astype(np.int64))
    return MessagePairs(pos[ia], pos[ib], tgt[ia].astype(np.int64))
