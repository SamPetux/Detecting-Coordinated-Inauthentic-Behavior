"""Layer fusion, Leiden communities and account scores (the no-GNN variant).

Input: per-layer edge tables with account ids `a`, `b` and a p-value (`p`) plus a significance flag (`sig`).
F1 (default): weight = sum over layers where the edge is significant of -log10(p).
F2: Fisher combination over all layers present, then BH across fused edges.
"""
from __future__ import annotations

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.stats import chi2

from cib.null import benjamini_hochberg

P_FLOOR = 1e-300


def fuse(layers: dict[str, pd.DataFrame], method: str = "sum_neglog10p", fdr_q: float = 0.01) -> pd.DataFrame:
    """Returns a, b, w and one -log10 p column per layer (NaN when the pair is absent from that layer)."""
    parts = []
    for name, e in layers.items():
        if e.empty:
            continue
        a, b = np.minimum(e.a.astype(str), e.b.astype(str)), np.maximum(e.a.astype(str), e.b.astype(str))
        nl = -np.log10(np.clip(e.p.to_numpy(dtype=float), P_FLOOR, 1))
        parts.append(pd.DataFrame({"a": a, "b": b, "layer": name, "nl": nl, "sig": e.sig.to_numpy(dtype=bool)}))
    if not parts:
        return pd.DataFrame(columns=["a", "b", "w"])
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index=["a", "b"], columns="layer", values="nl", aggfunc="max")
    sig = long.pivot_table(index=["a", "b"], columns="layer", values="sig", aggfunc="max").fillna(False).astype(bool)
    if method == "sum_neglog10p":
        w = wide.where(sig).sum(axis=1, min_count=1)
    elif method == "fisher":
        k = wide.notna().sum(axis=1)
        stat = 2 * np.log(10) * wide.fillna(0).sum(axis=1)
        p = chi2.sf(stat, 2 * k)
        keep = benjamini_hochberg(p) <= fdr_q
        w = pd.Series(np.where(keep, -np.log10(np.clip(p, P_FLOOR, 1)), np.nan), index=wide.index)
    else:
        raise ValueError(method)
    out = wide.add_prefix("nl_").assign(w=w).dropna(subset=["w"]).reset_index()
    return out


def leiden(edges: pd.DataFrame, weight: str = "w", resolution: float = 1.0, min_cluster: int = 3,
           seed: int = 0) -> pd.DataFrame:
    """account_id, cluster_id (-1 for clusters smaller than min_cluster), score = weighted degree."""
    if edges.empty:
        return pd.DataFrame(columns=["account_id", "cluster_id", "score"])
    names = pd.Index(pd.unique(np.r_[edges.a.to_numpy(), edges.b.to_numpy()]))
    g = ig.Graph(n=len(names), edges=list(zip(names.get_indexer(edges.a), names.get_indexer(edges.b))))
    g.es["weight"] = edges[weight].to_numpy(dtype=float)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed)
    memb = np.array(part.membership)
    size = np.bincount(memb)
    cid = np.where(size[memb] >= min_cluster, memb, -1)
    return pd.DataFrame({"account_id": names, "cluster_id": cid, "score": g.strength(weights="weight")})
