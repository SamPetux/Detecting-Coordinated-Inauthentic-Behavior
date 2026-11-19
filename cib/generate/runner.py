"""Run text requests for a plan (two rounds: comments, then in-campaign replies) and realise messages."""
from __future__ import annotations

import pandas as pd

from cib.generate import ladder
from cib.generate.llm import LLM

GEN_PARAMS = {"gpt-5-mini": {"reasoning": {"effort": "minimal"}}, "gpt-5-nano": {"reasoning": {"effort": "minimal"}}}


def _run(llm: LLM, req: pd.DataFrame, mode: str, tag: str) -> dict[str, str | None]:
    texts: dict[str, str | None] = {}
    for model, g in req.groupby("model"):
        params = GEN_PARAMS.get(model, {})
        if mode == "batch":
            out = llm.run_batch(model, list(zip(g.instructions, g.req_prompt)), params=params,
                                max_output_tokens=300, tag=tag, nonces=list(g.req_key))
        else:
            out = [llm.generate(model, i, p, params=params, max_output_tokens=300, tag=tag, nonce=k)
                   for i, p, k in zip(g.instructions, g.req_prompt, g.req_key)]
        texts.update(zip(g.req_key, out))
    return texts


def generate(plan: pd.DataFrame, llm: LLM, *, mode: str = "online", seed: int = 0, tag: str = "") -> pd.DataFrame:
    texts = _run(llm, ladder.requests(plan), mode, tag)
    # reply chains: each round can only answer parents generated in earlier rounds
    for _ in range(10):
        nxt = ladder.requests(plan, texts)
        nxt = nxt[~nxt.req_key.isin(texts.keys())]
        if nxt.empty:
            break
        texts.update(_run(llm, nxt, mode, tag))
    return ladder.realize(plan, texts, seed=seed)
