"""Generate all synthetic campaigns and benign-LLM controls for one split (Batch API, resumable via the cache).

Exploration: briefs of the explore half, generator A, seeds 1-2, every level; plus generator C (exploration-only
holdout) at L3 and benign. Confirmation (only after the protocol freeze): generators A and B, seeds from config.
Output: data/synthetic/<split>/messages.parquet and plan.parquet (prompts with real post texts are not saved).
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cib.generate import ladder, runner
from cib.generate.briefs import split_briefs
from cib.generate.llm import LLM

SIZES = [(5, 5), (10, 10), (20, 10), (40, 20), (10, 5), (20, 20)]
DELAYS = [300, 1800, 7200, 21600, 86400]
SHARES = [1.0, 0.5, 0.25]
EVASION = list(itertools.product(DELAYS, SHARES))


def plans_for(bg: pd.DataFrame, split: str, generators: list[str], seeds: list[int], levels: list[str],
              salt: str, tag: str) -> list[pd.DataFrame]:
    out = []
    briefs = split_briefs()[split]
    k = 0
    for seed in seeds:
        for bi, brief in enumerate(briefs):
            for level in levels:
                n, t = SIZES[(bi + seed) % len(SIZES)]
                delay, share = (1200, 1.0)
                hist = 4.0
                if level in ("L4", "L5"):
                    delay, share = EVASION[k % len(EVASION)]
                    k += 1
                if level == "L5":
                    hist = 12.0
                plan = ladder.plan_campaign(bg, brief, level=level, n_accounts=n, n_targets=t, salt=salt,
                                            seed=1000 * seed + bi, generators=generators, split=split,
                                            median_delay_s=delay, participation=share, history_lambda=hist)
                plan["campaign_id"] = plan.campaign_id + f"-{tag}"
                plan["row_id"] = plan.row_id.str.replace("#", f"-{tag}#", regex=False)
                plan["reply_to"] = plan.reply_to.str.replace("#", f"-{tag}#", regex=False)
                plan["req_key"] = plan.req_key + f"|{tag}"
                plan["account_id"] = plan.account_id.map(lambda a: ladder.schema.hash_account(f"{tag}:{a}", salt))
                plan["median_delay_s"], plan["participation"] = delay, share
                out.append(plan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="explore", choices=["explore", "confirm"])
    ap.add_argument("--mode", default="batch", choices=["batch", "online"])
    args = ap.parse_args()
    load_dotenv()
    salt = os.environ["CIB_SALT"]
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    # community accounts publish 96% of the posts: drop only their comments, never their posts
    bg = bg[(bg.split == args.split) & ((bg.kind == "post") | ~bg.author_is_community)]
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    if args.split == "explore":
        plans = plans_for(bg, "explore", ["gpt-4o-mini"], [1, 2], levels, salt, "A")
        plans += plans_for(bg, "explore", ["gpt-5-nano"], [3], ["L3"], salt, "C")
        plans.append(ladder.plan_benign(bg, n_accounts=200, salt=salt, seed=11, generators=["gpt-4o-mini"],
                                        split="explore"))
        plans.append(ladder.plan_benign(bg, n_accounts=50, salt=salt, seed=12, generators=["gpt-5-nano"],
                                        split="explore"))
    else:
        raise SystemExit("confirmatory generation is allowed only after the protocol freeze (T16)")
    plan = pd.concat(plans, ignore_index=True)
    llm = LLM()
    req = ladder.requests(plan)
    est = sum(llm.estimate(m, len(g), 1500, 300, batch=args.mode == "batch") for m, g in req.groupby("model"))
    print(f"plan rows {len(plan)}, first-round requests {len(req)}, estimate (first round, upper) {est:.2f} USD, "
          f"spent so far {llm.ledger.total():.2f} USD", flush=True)
    msgs = runner.generate(plan, llm, mode=args.mode, seed=0, tag=f"gen-{args.split}")
    out = Path(f"data/synthetic/{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    msgs.to_parquet(out / "messages.parquet", index=False)
    plan.drop(columns=["persona", "req_prompt"]).to_parquet(out / "plan.parquet", index=False)
    print(f"realised {len(msgs)} of {len(plan)} ({1 - len(msgs) / len(plan):.1%} dropped); "
          f"spent {llm.ledger.total():.2f} USD", flush=True)


if __name__ == "__main__":
    main()
