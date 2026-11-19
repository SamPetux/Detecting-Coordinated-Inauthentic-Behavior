"""Exploration probe: layers T and L with the BPAP null vs a global null on organic explore data only."""
import time, numpy as np, pandas as pd
from cib import null
from cib.layers import cotarget, lexical
from cib.layers.pairs import account_codes
df = pd.read_parquet("data/interim/vk_messages.parquet")
df = df[df.split == "explore"].reset_index(drop=True)
acc, names = account_codes(df.account_id)
mov = df.kind.ne("post").to_numpy()
for slot in (3600, 86400):
    b = null.buckets(df, slot)
    for name, fn in (("T600", lambda: cotarget.message_pairs(df, 600)), ("L0.6", lambda: lexical.message_pairs(df, 3600, 0.6))):
        t0 = time.time(); p = fn(); t1 = time.time()
        e = null.edge_significance(p, acc, len(names), b, mov, permutations=20, min_obs=2, fdr_q=0.01)
        g = null.edge_significance(p, acc, len(names), b, mov, permutations=20, min_obs=2, fdr_q=0.01, global_null=True)
        sig_acc = len(set(e.u[e.sig]) | set(e.v[e.sig]))
        print(f"slot={slot} {name}: pairs={len(p)} cand={len(e)} sig={int(e.sig.sum())} accounts_with_sig={sig_acc} "
              f"(of {len(names)}) | global-null sig={int(g.sig.sum())} | pairs {t1-t0:.0f}s null {time.time()-t1:.0f}s", flush=True)
