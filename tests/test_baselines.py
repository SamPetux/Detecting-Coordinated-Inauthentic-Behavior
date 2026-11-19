import numpy as np
import pandas as pd

from cib import baselines
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from test_layers_null import toy  # noqa: E402


def test_raw_baselines_rank_coordinated_first():
    df = toy()
    for score in (baselines.raw_cotarget(df, 120), baselines.copypaste(df)):
        top = score.sort_values(ascending=False).index[:3]
        assert set(top) == {"c1", "c2", "c3"}


def test_luceri_text_similarity_counts_pairs():
    df = toy()
    m = df[df.kind != "post"]
    emb = np.random.default_rng(0).normal(size=(len(m), 8))
    emb[m.account_id.str.startswith("c").to_numpy()] = 1.0
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    s = baselines.luceri_text_similarity(df, emb, pd.Series(np.arange(len(m)), index=m.msg_id), threshold=0.99)
    assert set(s.sort_values(ascending=False).index[:3]) == {"c1", "c2", "c3"}


def test_luceri_clean():
    assert baselines.luceri_clean("Привет, [id1|Олег]! 2026 https://x.ru 😊") == "привет"
