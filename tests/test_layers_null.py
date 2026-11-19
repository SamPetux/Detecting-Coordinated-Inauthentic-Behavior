import numpy as np
import pandas as pd

from cib import null
from cib.layers import cotarget, lexical, semantic
from cib.layers.pairs import MessagePairs, account_codes, account_counts, windowed_pairs


def toy(seed=0, n_org=60, n_posts=40, coord=("c1", "c2", "c3")):
    """Organic accounts comment random posts; three coordinated accounts hit 8 posts within a minute."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_posts):
        t0 = 1_000_000 + p * 7200
        rows.append(dict(msg_id=f"p{p}", account_id="g", ts=t0, kind="post", target_id=f"p{p}", community="g",
                         text="post"))
        for k in range(8):
            rows.append(dict(msg_id=f"p{p}o{k}", account_id=f"o{rng.integers(n_org)}", ts=t0 + int(rng.integers(0, 5400)),
                             kind="comment", target_id=f"p{p}", community="g", text=f"organic {p} {k} " * 3))
        if p < 8:
            for j, a in enumerate(coord):
                rows.append(dict(msg_id=f"p{p}{a}", account_id=a, ts=t0 + 300 + 10 * j, kind="comment",
                                 target_id=f"p{p}", community="g", text="платные парковки нужны городу срочно"))
    return pd.DataFrame(rows).reset_index(drop=True)


def test_windowed_pairs_respects_groups_and_window():
    ts = np.array([0, 5, 20, 0, 3])
    g = np.array([0, 0, 0, 1, 1])
    ia, ib = windowed_pairs(ts, 10, g)
    got = {tuple(sorted(x)) for x in zip(ia, ib)}
    assert got == {(0, 1), (3, 4)}


def test_account_counts_counts_episodes_once():
    acc = np.array([0, 1, 0, 1])
    pairs = MessagePairs(np.array([0, 2, 0]), np.array([1, 3, 3]), np.array([5, 5, 6]))
    c = account_counts(pairs, acc, 2)
    assert c.to_dict() == {1: 2}  # key 0*2+1, episodes {5, 6}


def test_permutation_keeps_bucket_composition():
    rng = np.random.default_rng(0)
    acc = rng.integers(0, 5, 200)
    bucket = rng.integers(0, 10, 200)
    movable = rng.random(200) > 0.2
    perm = null.permute_within(acc, bucket, movable, rng)
    assert (perm[~movable] == acc[~movable]).all()
    for b in range(10):
        m = (bucket == b) & movable
        assert sorted(perm[m]) == sorted(acc[m])


def test_cotarget_null_flags_only_coordinated():
    df = toy()
    acc, names = account_codes(df.account_id)
    pairs = cotarget.message_pairs(df, window_s=120)
    e = null.edge_significance(pairs, acc, len(names), null.buckets(df), df.kind.ne("post").to_numpy(),
                               permutations=50, min_obs=2, fdr_q=0.01)
    sig = {tuple(sorted((names[u], names[v]))) for u, v in zip(e.u[e.sig], e.v[e.sig])}
    assert sig == {("c1", "c2"), ("c1", "c3"), ("c2", "c3")}


def test_lexical_and_semantic_find_duplicates():
    df = toy()
    lp = lexical.message_pairs(df, window_s=3600, min_jaccard=0.9)
    acc, names = account_codes(df.account_id)
    c = account_counts(lp, acc, len(names))
    assert c.max() == 8
    ids = df.msg_id[df.kind != "post"]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(ids), 8))
    coord = df.set_index("msg_id").loc[ids].account_id.str.startswith("c").to_numpy()
    emb[coord] = 1.0
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sp = semantic.message_pairs(df, emb, pd.Series(np.arange(len(ids)), index=ids), window_s=3600, min_cos=0.99)
    c = account_counts(sp, acc, len(names))
    assert len(c) == 3 and c.min() == 8


def test_bh_monotone():
    q = null.benjamini_hochberg(np.array([0.001, 0.04, 0.03, 0.5]))
    assert np.all(q >= np.array([0.001, 0.04, 0.03, 0.5]) - 1e-12) and q[0] == 0.004
