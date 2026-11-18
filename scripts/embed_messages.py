"""Embed one split of a message table with the chosen model (resumable chunks under data/interim/emb/)."""
from __future__ import annotations

import argparse

import pandas as pd

from cib import embed

MODEL = "intfloat/multilingual-e5-small"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/interim/vk_messages.parquet")
    ap.add_argument("--split", required=True)
    ap.add_argument("--name", required=True, help="cache sub-directory, e.g. vk_explore")
    ap.add_argument("--luceri", action="store_true", help="Luceri et al. baseline: their model, cleaning, >4 words")
    args = ap.parse_args()
    df = pd.read_parquet(args.table)
    df = df[df.split == args.split]
    kw = {}
    model = MODEL
    if args.luceri:
        from cib import baselines
        model = baselines.LUCERI_MODEL
        kw = dict(preprocess=baselines.luceri_clean, keep=lambda t: len(t.split()) > 4)
    x, ids = embed.embed_messages(df, model, cache_dir=f"data/interim/emb/{args.name}",
                                  log=lambda s: print(s, flush=True), **kw)
    print(f"done: {x.shape} for {len(ids)} messages", flush=True)


if __name__ == "__main__":
    main()
