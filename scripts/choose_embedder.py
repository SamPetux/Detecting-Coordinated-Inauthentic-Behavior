"""T04: pick the embedding model on exploratory data only.

Positives: pairs of pilot campaign messages under the same target (paraphrases of one goal by different accounts).
Hard negatives: pairs of organic comments under the same post (same topic, independent authors).
Easy negatives: organic comments in the same community and hour.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cib import embed
from cib.text import sem_eligible

MODELS = ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "deepvk/USER-bge-m3"]
N_NEG = 3000
MAX_HOURS = 6.0
OUT = "docs/results/embedder_choice.csv"


def pairs_within(df: pd.DataFrame, key: str, rng: np.random.Generator, cap: int) -> list[tuple[str, str]]:
    out = []
    for _, g in df.groupby(key):
        ids = g.msg_id.tolist()
        if len(ids) > 1:
            out.extend((a, b) for a, b in itertools.combinations(ids, 2)
                       if g.account_id[g.msg_id == a].item() != g.account_id[g.msg_id == b].item())
    if len(out) > cap:
        out = [out[i] for i in rng.choice(len(out), cap, replace=False)]
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    syn = pd.read_parquet("data/synthetic/pilot/messages.parquet")
    syn = syn[syn.campaign_id.notna() & syn.text.map(sem_eligible)]
    syn["grp"] = syn.campaign_id + "|" + syn.target_id
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[(bg.split == "explore") & (bg.kind != "post") & ~bg.author_is_community & bg.text.map(sem_eligible)]
    sub = bg[bg.target_id.isin(bg.target_id.drop_duplicates().sample(800, random_state=0))]
    sub = sub.groupby("target_id").head(12)
    hour = bg.sample(8000, random_state=1).assign(h=lambda d: d.community + "|" + (d.ts // 3600).astype(str))

    pos = pairs_within(syn, "grp", rng, 10**6)
    hard = pairs_within(sub, "target_id", rng, N_NEG)
    easy = pairs_within(hour, "h", rng, N_NEG)
    texts = pd.concat([syn, sub, hour]).drop_duplicates("msg_id").set_index("msg_id").text
    print(f"pairs: positive {len(pos)}, hard negative {len(hard)}, easy negative {len(easy)}; texts {len(texts)}")

    rows = []
    clean = [embed.clean_for_similarity(t) for t in texts]
    for name in MODELS:
        model = embed.load_model(name)
        t0 = time.time()
        embed.encode(model, name, clean[:500])
        rate = 500 / (time.time() - t0)
        hours = 415_000 / rate / 3600
        if hours > MAX_HOURS:
            rows.append({"model": name, "msgs_per_s": round(rate, 1), "hours_full_vk": round(hours, 2)})
            print(rows[-1], "-> too slow, quality not measured", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            continue
        x = embed.encode(model, name, clean)
        e = dict(zip(texts.index, x))
        cos = lambda ps: np.array([float(e[a] @ e[b]) for a, b in ps])
        cp, ch, ce = cos(pos), cos(hard), cos(easy)
        y = lambda neg: np.r_[np.ones(len(cp)), np.zeros(len(neg))]
        rows.append({"model": name, "msgs_per_s": round(rate, 1),
                     "hours_full_vk": round(hours, 2),
                     "auc_vs_same_post": round(roc_auc_score(y(ch), np.r_[cp, ch]), 3),
                     "auc_vs_same_hour": round(roc_auc_score(y(ce), np.r_[cp, ce]), 3),
                     "cos_pos_median": round(float(np.median(cp)), 3),
                     "cos_hard_p99": round(float(np.quantile(ch, .99)), 3),
                     "recall_pos_at_hard_p99": round(float((cp > np.quantile(ch, .99)).mean()), 3)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
