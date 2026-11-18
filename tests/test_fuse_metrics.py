import numpy as np
import pandas as pd

from cib import fuse, metrics


def clique(names, p=1e-6, sig=True):
    return [(a, b, p, sig) for i, a in enumerate(names) for b in names[i + 1:]]


def test_fuse_and_leiden_recover_two_cliques():
    t = pd.DataFrame(clique(list("abcde")) + [("e", "x", 0.5, False)], columns=["a", "b", "p", "sig"])
    s = pd.DataFrame(clique(list("vwxyz")), columns=["a", "b", "p", "sig"])
    f = fuse.fuse({"T": t, "S": s})
    assert ("e", "x") not in set(zip(f.a, f.b))  # non-significant edge does not enter F1
    cl = fuse.leiden(f, min_cluster=3)
    groups = cl[cl.cluster_id >= 0].groupby("cluster_id").account_id.apply(frozenset).tolist()
    assert set(groups) == {frozenset("abcde"), frozenset("vwxyz")}


def test_fisher_variant_runs():
    t = pd.DataFrame(clique(list("abc"), p=1e-4), columns=["a", "b", "p", "sig"])
    f = fuse.fuse({"T": t, "S": t.assign(p=1e-3)}, method="fisher")
    assert len(f) == 3 and (f.w > 4).all()


def test_recall_at_fpr_threshold_from_organic():
    idx = [f"o{i}" for i in range(1000)] + [f"c{i}" for i in range(10)]
    score = pd.Series(np.r_[np.linspace(0, 1, 1000), np.full(10, 2.0)], index=idx)
    pos = pd.Series([i.startswith("c") for i in idx], index=idx)
    rec, thr = metrics.recall_at_fpr(score, pos, ~pos, 0.001)
    assert rec == 1.0 and metrics.flagged_rate(score, pd.Index(idx[:1000]), thr) <= 0.001


def test_best_cluster_f1_and_nmi():
    cl = pd.DataFrame({"account_id": ["a", "b", "c", "d"], "cluster_id": [0, 0, 0, -1], "score": 1.0})
    camp = pd.Series({"a": "k", "b": "k", "c": None, "d": "k", "e": None})
    f1 = metrics.best_cluster_f1(cl, camp)
    assert abs(f1["k"] - 2 * (2 / 3) * (2 / 3) / (4 / 3)) < 1e-9
    assert 0 <= metrics.nmi(cl, camp) <= 1


def test_stats_helpers():
    from cib import stats
    v = pd.Series(np.r_[np.ones(10), np.zeros(10)])
    est, lo, hi = stats.cluster_bootstrap(v, pd.Series(np.repeat(np.arange(10), 2)), n=500)
    assert lo <= est <= hi
    d, p = stats.paired_wilcoxon(np.arange(20) + 1.0, np.arange(20))
    assert d == 1.0 and p < 0.01
    h = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == {"a": 0.03, "c": 0.06, "b": 0.06}
