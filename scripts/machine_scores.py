"""Cross-fitted machine-generation probabilities for the exploratory split (layer M, detector CharNgramLR).

Synthetic training texts come from generator A only. Folds: seed 1 vs seed 2 of the campaigns, and two halves of
the organic accounts; every message is scored by a model that never saw it (nor its account). Generator-C messages
are scored by both models and averaged — they are the exploration-only transfer check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cib.layers.machine import CharNgramLR

N_ORG_TRAIN = 40000


def main() -> None:
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[(bg.split == "explore") & (bg.kind != "post") & bg.text.str.len().gt(0) & ~bg.author_is_community]
    syn = pd.concat([pd.read_parquet("data/synthetic/explore/messages.parquet"),
                     pd.read_parquet("data/synthetic/explore_controls/benign_hot.parquet")], ignore_index=True)
    syn = syn[syn.text.str.len() > 0]
    tag = syn.campaign_id.fillna("").str.extract(r"-(A|C)$")[0]
    seed = syn.campaign_id.fillna("").str.extract(r"-s(\d)\d*-")[0]
    # generator C and the hot-thread control are never used for training: both models score them
    is_c = (tag == "C") | syn.msg_id.str.contains("benign-explore-s12") | (syn.level == "benign_hot")
    benign_a = syn.msg_id.str.contains("benign-explore-s11")
    accs = bg.account_id.drop_duplicates().sample(frac=1, random_state=0)
    half = pd.Series(np.arange(len(accs)) % 2, index=accs.to_numpy())
    org_fold = bg.account_id.map(half)
    out = []
    for k in (0, 1):
        s_train = syn[(~is_c) & ((seed == str(k + 1)) | (benign_a & (syn.msg_id.str.len() % 2 == k)))]
        o_train = bg[org_fold == k].sample(min(N_ORG_TRAIN, int((org_fold == k).sum())), random_state=k)
        model = CharNgramLR(seed=k).fit(list(s_train.text) + list(o_train.text),
                                        np.r_[np.ones(len(s_train)), np.zeros(len(o_train))])
        s_test = syn[(~is_c) & ((seed == str(2 - k)) | (benign_a & (syn.msg_id.str.len() % 2 != k)))]
        o_test = bg[org_fold != k]
        for part in (s_test, o_test):
            out.append(pd.Series(model.predict_proba(list(part.text)), index=part.msg_id))
        out.append(pd.Series(model.predict_proba(list(syn[is_c].text)), index=syn[is_c].msg_id) / 2)
    prob = pd.concat(out).groupby(level=0).sum().clip(0, 1)
    prob.rename("p").to_frame().to_parquet("data/interim/machine_prob_explore.parquet")
    org_p = prob.reindex(bg.msg_id).dropna()
    for name, mask in (("generator A (held-out seed)", ~is_c), ("generator C (unseen)", is_c)):
        sp = prob.reindex(syn[mask].msg_id).dropna()
        auc = roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(org_p))], np.r_[sp, org_p])
        print(f"{name}: messages {len(sp)}, AUC vs organic {auc:.3f}, median p {sp.median():.2f}")
    print(f"organic median p {org_p.median():.3f}, p90 {org_p.quantile(.9):.3f}")


if __name__ == "__main__":
    main()
