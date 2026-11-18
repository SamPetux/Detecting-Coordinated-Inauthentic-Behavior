import numpy as np
import pandas as pd

from cib import gnn


def synthetic_graph(seed=0, n_org=200, n_camp=20):
    rng = np.random.default_rng(seed)
    org = [f"o{i}" for i in range(n_org)]
    camp = [f"c{i}" for i in range(n_camp)]
    rows = []
    for _ in range(600):
        a, b = rng.choice(org, 2, replace=False)
        rows.append((a, b, rng.exponential(0.5), np.nan, np.nan))
    for i, a in enumerate(camp):
        for b in camp[i + 1:]:
            if rng.random() < 0.5:
                rows.append((a, b, rng.exponential(0.5) + 2, rng.exponential(0.5) + 2, np.nan))
    e = pd.DataFrame(rows, columns=["a", "b", "nl_T", "nl_S", "nl_L"])
    e["w"] = e[["nl_T", "nl_S"]].sum(axis=1)
    y = pd.Series({**{a: 0 for a in org}, **{c: 1 for c in camp}})
    return e, y


def test_gnn_separates_planted_group():
    e, y = synthetic_graph()
    d, names = gnn.build_graph(e, ["nl_T", "nl_S", "nl_L"], pd.DataFrame(index=y.index))
    yy = y.reindex(names).to_numpy()
    rng = np.random.default_rng(1)
    train = rng.random(len(names)) < 0.6
    model = gnn.train(d, yy, train, val_mask=~train, epochs=200, seed=0)
    prob = gnn.predict(model, d, names)
    test = ~train
    assert prob[names[test & (yy == 1)]].mean() > prob[names[test & (yy == 0)]].mean() + 0.3
    cl = gnn.groups(e, prob, theta=0.5)
    assert cl.cluster_id.max() >= 0
