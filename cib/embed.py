"""Local sentence embeddings with an on-disk cache keyed by msg_id."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cib.text import clean_for_similarity

PREFIX = {"intfloat/multilingual-e5-small": "query: ", "intfloat/multilingual-e5-base": "query: "}


def load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def encode(model, name: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    pre = PREFIX.get(name, "")
    return model.encode([pre + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def embed_messages(msgs: pd.DataFrame, name: str, cache_dir: str = "data/interim/emb", chunk: int = 20000,
                   log=print, preprocess=clean_for_similarity, keep=None) -> tuple[np.ndarray, pd.Index]:
    """Embeddings for every message with text; resumable: finished chunks are kept as separate .npy files.

    `keep(clean_text) -> bool` filters messages after preprocessing (e.g. the >4-word rule of a baseline)."""
    d = Path(cache_dir) / name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    m = msgs[msgs.text.str.len() > 0]
    texts = [preprocess(t) for t in m.text]
    if keep is not None:
        mask = np.array([keep(t) for t in texts], bool)
        m, texts = m[mask], [t for t, k in zip(texts, mask) if k]
    ids = pd.Index(m.msg_id)
    index_file = d / "index.parquet"
    # chunks are positional: reusing them for a different message list would silently misalign vectors
    if index_file.exists() and not pd.read_parquet(index_file).msg_id.equals(pd.Series(ids, name="msg_id")):
        raise ValueError(f"{d} was built for another message list; use a new cache_dir")
    pd.Series(ids, name="msg_id").to_frame().to_parquet(index_file, index=False)
    model = None
    parts = []
    for start in range(0, len(texts), chunk):
        f = d / f"part_{start:08d}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        model = model or load_model(name)
        x = encode(model, name, texts[start:start + chunk])
        np.save(f.with_suffix(".tmp.npy"), x)
        os.replace(f.with_suffix(".tmp.npy"), f)
        parts.append(x)
        log(f"{name}: {min(start + chunk, len(texts))}/{len(texts)}")
    return np.vstack(parts) if parts else np.zeros((0, 0), np.float32), ids
