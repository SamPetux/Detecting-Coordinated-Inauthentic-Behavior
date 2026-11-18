"""Layer Y: account writing-style similarity.

Account vector = standardised handcrafted rates + SVD of the character 3-gram TF-IDF centroid. Edges come from
k nearest neighbours; their p-value is the tail share among random pairs of the same accounts (an empirical null:
style has no time dimension, so the bucket permutation does not apply).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def handcrafted(texts: pd.Series) -> pd.DataFrame:
    t = texts.fillna("")
    n = t.str.len().clip(lower=1)
    return pd.DataFrame({
        "log_len": np.log1p(t.str.len()),
        "punct": t.str.count(r"[.,;:]") / n,
        "excl": t.str.contains("!").astype(float),
        "quest": t.str.contains(r"\?").astype(float),
        "ellipsis": t.str.contains(r"\.\.\.|…").astype(float),
        "emoji": t.map(lambda s: len(EMOJI.findall(s))) / n,
        "upper_start": t.str.match(r"^[A-ZА-ЯЁ]").astype(float),
        "caps": t.str.count(r"[A-ZА-ЯЁ]") / n,
        "dash": t.str.contains("—|–").astype(float),
        "no_final_punct": (~t.str.contains(r"[.!?…)]\s*$")).astype(float),
    })


def account_vectors(msgs: pd.DataFrame, min_messages: int = 3, svd_dim: int = 64, seed: int = 0
                    ) -> tuple[np.ndarray, pd.Index]:
    m = msgs[msgs.kind.ne("post") & msgs.text.str.len().gt(0)]
    keep = m.account_id.map(m.account_id.value_counts()) >= min_messages
    m = m[keep]
    hand = handcrafted(m.text).groupby(m.account_id.to_numpy()).mean()
    tf = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5, sublinear_tf=True, max_features=100_000)
    X = tf.fit_transform(m.text.str.lower())
    acc_codes, accs = pd.factorize(m.account_id)
    import scipy.sparse as sp
    agg = sp.csr_matrix((np.ones(len(acc_codes)), (acc_codes, np.arange(len(acc_codes)))),
                        shape=(len(accs), len(acc_codes)))
    centroid = normalize(agg @ X)
    k = min(svd_dim, centroid.shape[1] - 1)
    z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(centroid)
    h = StandardScaler().fit_transform(hand.reindex(accs).to_numpy())
    v = normalize(np.hstack([h, StandardScaler().fit_transform(z)]))
    return v.astype(np.float32), pd.Index(accs)


def style_edges(v: np.ndarray, accs: pd.Index, k: int = 20, n_null: int = 500_000, seed: int = 0) -> pd.DataFrame:
    """kNN edges with a p-value against the cosine distribution of random account pairs (normal on Fisher-z)."""
    if len(accs) < 3:
        return pd.DataFrame(columns=["a", "b", "cos", "p"])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(accs)), metric="cosine").fit(v)
    dist, idx = nn.kneighbors(v)
    rows = [(i, j, 1 - d) for i in range(len(accs)) for j, d in zip(idx[i, 1:], dist[i, 1:]) if i < j]
    e = pd.DataFrame(rows, columns=["i", "j", "cos"]).drop_duplicates(["i", "j"])
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(accs), n_null), rng.integers(0, len(accs), n_null)
    keep = a != b
    null = np.arctanh(np.clip(np.einsum("ij,ij->i", v[a[keep]], v[b[keep]]), -0.999999, 0.999999))
    # parametric tail on Fisher-z: an empirical tail cannot reach the p-values that BH over all pairs requires
    z = (np.arctanh(np.clip(e.cos.to_numpy(), -0.999999, 0.999999)) - null.mean()) / null.std()
    e["p"] = norm.sf(z)
    e["a"], e["b"] = accs[e.i.to_numpy()], accs[e.j.to_numpy()]
    return e[["a", "b", "cos", "p"]]


def pair_edges(v: np.ndarray, accs: pd.Index, pairs: pd.DataFrame, n_null: int = 500_000, seed: int = 0
               ) -> pd.DataFrame:
    """Style evidence for given account pairs (a, b): cosine and its p-value against random account pairs.

    Used to reinforce pairs already linked by synchronisation (T or S); on its own, style similarity mostly links
    organic users who write alike."""
    pos = pd.Series(np.arange(len(accs)), index=accs)
    ok = pairs.a.isin(pos.index) & pairs.b.isin(pos.index)
    p = pairs[ok]
    if p.empty:
        return pd.DataFrame(columns=["a", "b", "cos", "p", "sig"])
    ia, ib = pos.loc[p.a].to_numpy(), pos.loc[p.b].to_numpy()
    cos = np.einsum("ij,ij->i", v[ia], v[ib])
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(accs), n_null), rng.integers(0, len(accs), n_null)
    keep = a != b
    null = np.arctanh(np.clip(np.einsum("ij,ij->i", v[a[keep]], v[b[keep]]), -0.999999, 0.999999))
    z = (np.arctanh(np.clip(cos, -0.999999, 0.999999)) - null.mean()) / null.std()
    pv = norm.sf(z)
    return pd.DataFrame({"a": p.a.to_numpy(), "b": p.b.to_numpy(), "cos": cos, "p": pv, "sig": pv <= 0.01})
