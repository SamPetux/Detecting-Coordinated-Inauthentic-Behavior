"""Collect finished Batch API results that a killed process never downloaded: cache texts and log their cost.

Only batches created after `--since` (unix time) are touched, so batches of other projects on the same key are
skipped. Results already in the cache are not billed twice in the ledger.
"""
from __future__ import annotations

import argparse
import json
import re

from dotenv import load_dotenv

from cib.generate.llm import LLM


def price_key(model: str, known: set[str]) -> str:
    """'gpt-4o-mini-2024-07-18' -> 'gpt-4o-mini'."""
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if base not in known:
        raise KeyError(f"no price for {model}")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, required=True)
    args = ap.parse_args()
    load_dotenv()
    llm = LLM()
    for b in llm.client.batches.list(limit=100):
        if b.created_at < args.since or not b.output_file_id or b.status not in ("completed", "expired", "cancelled"):
            continue
        new = skipped = 0
        for line in llm.client.files.content(b.output_file_id).text.splitlines():
            o = json.loads(line)
            if llm.cache.get(o["custom_id"]) is not None:
                skipped += 1
                continue
            body = (o.get("response") or {}).get("body") or {}
            if not body:
                continue
            llm._absorb_batch_line(o, price_key(body["model"], set(llm.prices)), tag=f"recovered:{b.id}")
            new += 1
        print(f"{b.id} {b.status}: absorbed {new}, already cached {skipped}", flush=True)
    print(f"ledger total {llm.ledger.total():.2f} USD")


if __name__ == "__main__":
    main()
