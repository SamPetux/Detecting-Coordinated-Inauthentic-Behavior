"""Exploration control for H5: independent LLM users concentrated on the same busy threads (online, small)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cib.generate import ladder, runner
from cib.generate.llm import LLM


def main() -> None:
    load_dotenv()
    salt = os.environ["CIB_SALT"]
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    # community accounts publish 96% of the posts: drop only their comments, never their posts
    bg = bg[(bg.split == "explore") & ((bg.kind == "post") | ~bg.author_is_community)]
    plan = ladder.plan_benign_hot(bg, n_accounts=60, n_posts=15, salt=salt, seed=13, generators=["gpt-4o-mini"],
                                  split="explore")
    msgs = runner.generate(plan, LLM(), mode="online", tag="benign-hot")
    out = Path("data/synthetic/explore_controls")
    out.mkdir(parents=True, exist_ok=True)
    msgs.to_parquet(out / "benign_hot.parquet", index=False)
    print(f"benign_hot: {len(msgs)} messages, {msgs.account_id.nunique()} accounts, {msgs.target_id.nunique()} posts")


if __name__ == "__main__":
    main()
