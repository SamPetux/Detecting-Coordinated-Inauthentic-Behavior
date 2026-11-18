"""Exploration on BotSim-24 (explore split only): do the LLM bots show detectable coordination at all?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from cib import baselines, embed, evaluate, pipeline


def main() -> None:
    df = pd.read_parquet("data/interim/botsim_messages.parquet")
    df = df[df.split == "explore"].reset_index(drop=True)
    # bots do not author posts that humans comment on; keep their posts only as targets
    x, ids = embed.embed_messages(df, "intfloat/multilingual-e5-small", cache_dir="data/interim/emb/botsim_explore")
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    cfg = yaml.safe_load(open("configs/explore.yaml"))
    cfg["layers"]["style"]["enabled"] = True
    rows = []
    for bucket in ("community_hour", "community_day"):
        cfg["null_model"]["bucket"] = bucket
        res = pipeline.run(df, cfg, emb=x, row_of=row_of, log=lambda s: print("  ", s, flush=True))
        lab = evaluate.labels(df)
        scores = {"full": res.score, "full_noMY": pipeline.evidence(res.layers, ("T", "S", "L")),
                  "layer_T": evaluate.layer_score(res.layers["T"]), "layer_S": evaluate.layer_score(res.layers["S"]),
                  "raw_T": evaluate.raw_layer_score(res, "T", df), "raw_S": evaluate.raw_layer_score(res, "S", df),
                  "copypaste": baselines.copypaste(df)}
        t = evaluate.score_table(scores, lab)
        t.insert(0, "bucket", bucket)
        rows.append(t)
        print(bucket, t[["detector", "n_pos", "recall@0.001", "recall@0.01", "auprc"]].round(3).to_string(index=False),
              flush=True)
    pd.concat(rows).to_csv("docs/results/explore_botsim.csv", index=False)


if __name__ == "__main__":
    main()
