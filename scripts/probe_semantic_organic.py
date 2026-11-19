"""Exploration probe: layer S on organic explore data for several cosine thresholds, BPAP vs global null."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from cib import embed, null
from cib.layers import semantic
from cib.layers.pairs import account_codes


def main() -> None:
    df = pd.read_parquet("data/interim/vk_messages.parquet")
    df = df[df.split == "explore"].reset_index(drop=True)
    x, ids = embed.embed_messages(df, "intfloat/multilingual-e5-small", cache_dir="data/interim/emb/vk_explore")
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    eligible = df.text.map(lambda t: len(embed.clean_for_similarity(t)) >= 20)
    row_of = row_of[row_of.index.isin(df.msg_id[eligible])]
    acc, names = account_codes(df.account_id)
    mov = df.kind.ne("post").to_numpy()
    b = null.buckets(df, 3600)
    for tau in (0.90, 0.92, 0.94):
        t0 = time.time()
        p = semantic.message_pairs(df, x, row_of, window_s=3600, min_cos=tau)
        e = null.edge_significance(p, acc, len(names), b, mov, permutations=20)
        g = null.edge_significance(p, acc, len(names), b, mov, permutations=20, global_null=True)
        sig_acc = len(set(e.u[e.sig]) | set(e.v[e.sig]))
        print(f"tau={tau}: pairs={len(p)} cand={len(e)} sig={int(e.sig.sum())} accounts={sig_acc} | "
              f"global sig={int(g.sig.sum())} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
