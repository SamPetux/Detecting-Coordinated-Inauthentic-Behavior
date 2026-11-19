import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from test_layers_null import toy  # noqa: E402

from cib import pipeline  # noqa: E402


def test_pipeline_end_to_end():
    df = toy()
    cfg = yaml.safe_load(open("configs/explore.yaml"))
    cfg["layers"]["cotarget"]["window_s"] = 120
    cfg["layers"]["style"]["enabled"] = False
    m = df[df.kind != "post"]
    emb = np.random.default_rng(0).normal(size=(len(m), 8))
    emb[m.account_id.str.startswith("c").to_numpy()] = 1.0
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    prob = pd.Series(np.where(m.account_id.str.startswith("c"), 0.9, 0.1), index=m.msg_id)
    r = pipeline.run(df, cfg, emb=emb, row_of=pd.Series(np.arange(len(m)), index=m.msg_id), machine_prob=prob,
                     log=lambda *_: None)
    top = r.score.sort_values(ascending=False).index[:3]
    assert set(top) == {"c1", "c2", "c3"}
    assert {"T", "L", "S", "M"} <= set(r.layers)
