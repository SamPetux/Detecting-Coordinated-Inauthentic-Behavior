"""T06 pilot: 2 exploratory briefs x {L2, L3} + benign LLM users, generator A, online calls.

Writes data/synthetic/pilot/*.parquet and prints quality stats (drop rate, length vs organic comments).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cib.generate import ladder, runner
from cib.generate.briefs import split_briefs
from cib.generate.llm import LLM

GEN_A = "gpt-4o-mini"


def main() -> None:
    load_dotenv()
    salt = os.environ["CIB_SALT"]
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    # community accounts publish 96% of the posts: drop only their comments, never their posts
    bg = bg[(bg.split == "explore") & ((bg.kind == "post") | ~bg.author_is_community)]
    llm = LLM()
    out = Path("data/synthetic/pilot")
    out.mkdir(parents=True, exist_ok=True)
    plans, msgs = [], []
    for i, brief in enumerate(split_briefs()["explore"][:2]):
        for level in ("L2", "L3"):
            plan = ladder.plan_campaign(bg, brief, level=level, n_accounts=10, n_targets=5, salt=salt, seed=100 + i,
                                        generators=[GEN_A], split="explore")
            plans.append(plan)
            msgs.append(runner.generate(plan, llm, seed=100 + i, tag="pilot"))
    benign = ladder.plan_benign(bg, n_accounts=10, salt=salt, seed=200, generators=[GEN_A], split="explore",
                                max_msgs=6)
    plans.append(benign)
    msgs.append(runner.generate(benign, llm, seed=200, tag="pilot"))
    plan, m = pd.concat(plans, ignore_index=True), pd.concat(msgs, ignore_index=True)
    m.to_parquet(out / "messages.parquet", index=False)
    plan.drop(columns=["persona", "req_prompt"]).to_parquet(out / "plan.parquet", index=False)

    org_len = bg[bg.kind != "post"].text.str.len()
    m_len = m.text.str.len()
    print(f"planned {len(plan)}, realised {len(m)}, dropped {1 - len(m) / len(plan):.1%}")
    print(f"length median: organic {org_len.median():.0f}, synthetic {m_len.median():.0f}; "
          f"p90 organic {org_len.quantile(.9):.0f}, synthetic {m_len.quantile(.9):.0f}")
    print(m.groupby("level").agg(n=("text", "size"), len_med=("text", lambda s: s.str.len().median()),
                                 replies=("kind", lambda s: (s == "reply").sum())))
    print(f"spent so far: {llm.ledger.total():.4f} USD")


if __name__ == "__main__":
    main()
