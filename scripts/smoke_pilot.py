"""Exploration smoke test on the pilot campaigns: bucket and account-aggregator variants (not a result)."""
import numpy as np, pandas as pd, yaml
from cib import embed, evaluate
bg=pd.read_parquet("data/interim/vk_messages.parquet"); bg=bg[bg.split=="explore"].reset_index(drop=True)
syn=pd.read_parquet("data/synthetic/pilot/messages.parquet")
xb,ib=embed.embed_messages(bg,"intfloat/multilingual-e5-small",cache_dir="data/interim/emb/vk_explore")
xs,is_=embed.embed_messages(syn,"intfloat/multilingual-e5-small",cache_dir="data/interim/emb/syn_pilot",log=lambda s:None)
x=np.vstack([xb,xs]); row_of=pd.Series(np.arange(len(x)),index=ib.append(is_))
for lvl in ("L2","L3"):
  sub=syn[(syn.level==lvl)|(syn.level=="benign_llm")]
  for bucket in ("community_hour","community_day"):
    for agg in ("max","top3"):
      cfg=yaml.safe_load(open("configs/explore.yaml")); cfg["null_model"]["bucket"]=bucket; cfg["null_model"]["permutations"]=30; cfg["fusion"]["account_agg"]=agg
      table,res,lab=evaluate.run_once(bg,sub,cfg,emb=x,row_of=row_of,machine_prob=None,log=lambda s: None)
      t=table.set_index("detector").loc[["full","full_noM","full_noMY","layer_T","layer_S","raw_S","copypaste"],["recall@0.001","recall@0.01","benign_flag@0.01"]].round(2)
      print(lvl,bucket,agg, t.T.to_dict("list"), flush=True)
