"""Layer M for confirmation: CharNgramLR trained on all exploratory synthetic texts (A, C, controls) vs exploratory
organic comments; applied unchanged to confirmatory organic and synthetic messages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cib.layers.machine import CharNgramLR
from scripts.explore_runs import load_synthetic


def main() -> None:
    vk = pd.read_parquet("data/interim/vk_messages.parquet")
    org = vk[(vk.split == "explore") & (vk.kind != "post") & vk.text.str.len().gt(0) & ~vk.author_is_community]
    syn = load_synthetic()
    syn = syn[syn.text.str.len() > 0]
    org = org.sample(min(80000, len(org)), random_state=0)
    model = CharNgramLR(seed=0).fit(list(syn.text) + list(org.text), np.r_[np.ones(len(syn)), np.zeros(len(org))])
    test = pd.concat([vk[(vk.split == "confirm") & (vk.kind != "post") & vk.text.str.len().gt(0)],
                      pd.read_parquet("data/synthetic/confirm/messages.parquet")])
    test = test[test.text.str.len() > 0]
    p = pd.Series(model.predict_proba(list(test.text)), index=test.msg_id, name="p")
    p.to_frame().to_parquet("data/interim/machine_prob_confirm.parquet")
    print(f"scored {len(p)} confirmatory messages", flush=True)


if __name__ == "__main__":
    main()
