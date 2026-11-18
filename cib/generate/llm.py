"""OpenAI client with an SQLite cache, a cost ledger and a hard budget stop.

Every paid call in the project goes through `LLM.generate` (online) or `LLM.run_batch` (Batch API, half price).
Cache key = sha256 of the full request, so re-running a script never pays twice.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


class BudgetExceeded(RuntimeError):
    pass


def _chunks(lines: list[str], max_bytes: int) -> list[list[str]]:
    """Split JSONL lines so that every batch input file stays under the API size limit."""
    out, cur, size = [], [], 0
    for line in lines:
        n = len(line.encode()) + 1
        if cur and size + n > max_bytes:
            out.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += n
    if cur:
        out.append(cur)
    return out


@dataclass(frozen=True)
class Price:
    input: float          # USD per 1M tokens
    cached_input: float
    output: float


def load_prices(path: str = "configs/prices.yaml") -> dict[str, Price]:
    raw = yaml.safe_load(open(path))["models"]
    return {m: Price(**p) for m, p in raw.items()}


def request_key(model: str, instructions: str, prompt: str, params: dict, nonce: str = "") -> str:
    """`nonce` separates independent calls with identical prompts (e.g. two bots answering the same post)."""
    blob = json.dumps([model, instructions, prompt, params, nonce], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Ledger:
    """Append-only CSV of every billed request; the running total is the budget source of truth."""

    FIELDS = ["time", "model", "mode", "input_tokens", "cached_tokens", "output_tokens", "usd", "tag"]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def total(self) -> float:
        with self.path.open() as f:
            return sum(float(r["usd"]) for r in csv.DictReader(f))

    def add(self, model: str, mode: str, inp: int, cached: int, out: int, usd: float, tag: str) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), model, mode,
                                    inp, cached, out, f"{usd:.6f}", tag])


class Cache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, text TEXT, meta TEXT)")

    def get(self, k: str) -> str | None:
        row = self.db.execute("SELECT text FROM r WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, text: str, meta: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO r VALUES (?,?,?)", (k, text, json.dumps(meta)))
        self.db.commit()


class LLM:
    def __init__(self, client=None, prices: dict[str, Price] | None = None, *,
                 cache_path: str = "data/interim/llm_cache.sqlite", ledger_path: str = "results/api_costs.csv",
                 budget_usd: float | None = None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client
        self.prices = prices if prices is not None else load_prices()
        self.cache = Cache(cache_path)
        self.ledger = Ledger(ledger_path)
        self.budget = budget_usd if budget_usd is not None else float(os.environ.get("BUDGET_STOP_USD", 40))

    def cost(self, model: str, inp: int, cached: int, out: int, batch: bool = False) -> float:
        p = self.prices[model]
        usd = ((inp - cached) * p.input + cached * p.cached_input + out * p.output) / 1e6
        return usd / 2 if batch else usd

    def _guard(self, estimate: float) -> None:
        spent = self.ledger.total()
        if spent + estimate > self.budget:
            raise BudgetExceeded(f"spent {spent:.2f} + estimate {estimate:.2f} > stop {self.budget:.2f} USD")

    def estimate(self, model: str, n: int, inp_tokens: int, out_tokens: int, batch: bool) -> float:
        """Dry-run cost for n requests of the given average size; write it to the log before mass runs."""
        return n * self.cost(model, inp_tokens, 0, out_tokens, batch)

    def generate(self, model: str, instructions: str, prompt: str, *, params: dict | None = None,
                 max_output_tokens: int = 300, tag: str = "", est_input_tokens: int = 1500, nonce: str = "") -> str:
        params = dict(params or {}, max_output_tokens=max_output_tokens)
        k = request_key(model, instructions, prompt, params, nonce)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        self._guard(self.cost(model, est_input_tokens, 0, max_output_tokens))
        r = self.client.responses.create(model=model, instructions=instructions, input=prompt, **params)
        u = r.usage
        cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
        usd = self.cost(model, u.input_tokens, cached, u.output_tokens)
        self.ledger.add(model, "online", u.input_tokens, cached, u.output_tokens, usd, tag)
        self.cache.put(k, r.output_text, {"model": model, "usd": usd})
        return r.output_text

    def run_batch(self, model: str, requests: list[tuple[str, str]], *, params: dict | None = None,
                  max_output_tokens: int = 300, tag: str = "", poll_s: int = 600,
                  est_input_tokens: int = 1500, nonces: list[str] | None = None,
                  max_bytes: int = 150 * 1024 * 1024) -> list[str | None]:
        """requests = [(instructions, prompt)]; returns texts in order (None for failed items)."""
        params = dict(params or {}, max_output_tokens=max_output_tokens)
        nonces = nonces or [""] * len(requests)
        keys = [request_key(model, i, p, params, n) for (i, p), n in zip(requests, nonces)]
        todo = [(k, i, p) for k, (i, p) in zip(keys, requests) if self.cache.get(k) is None]
        # duplicate keys inside one batch would be paid twice; keep the first
        todo = list({k: (k, i, p) for k, i, p in todo}.values())
        if todo:
            self._guard(len(todo) * self.cost(model, est_input_tokens, 0, max_output_tokens, batch=True))
            lines = [json.dumps({"custom_id": k, "method": "POST", "url": "/v1/responses",
                                 "body": {"model": model, "instructions": i, "input": p, **params}},
                                ensure_ascii=False) for k, i, p in todo]
            for chunk in _chunks(lines, max_bytes):
                self._submit_and_collect(chunk, model, tag, poll_s)
        return [self.cache.get(k) for k in keys]

    def _submit_and_collect(self, lines: list[str], model: str, tag: str, poll_s: int) -> None:
        f = self.client.files.create(file=("batch.jsonl", io.BytesIO("\n".join(lines).encode())), purpose="batch")
        b = self.client.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h")
        while b.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(poll_s)
            b = self.client.batches.retrieve(b.id)
        if b.status == "failed":
            raise RuntimeError(f"batch {b.id} failed: {b.errors}")
        # expired/cancelled batches still return the finished part; missing items stay uncached and come back as None
        if b.output_file_id:
            for line in self.client.files.content(b.output_file_id).text.splitlines():
                self._absorb_batch_line(json.loads(line), model, tag)

    def _absorb_batch_line(self, o: dict, model: str, tag: str) -> None:
        resp = (o.get("response") or {})
        if resp.get("status_code") != 200:
            return
        body = resp["body"]
        text = "".join(c.get("text", "") for item in body.get("output", []) if item.get("type") == "message"
                       for c in item.get("content", []) if c.get("type") == "output_text")
        u = body["usage"]
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        usd = self.cost(model, u["input_tokens"], cached, u["output_tokens"], batch=True)
        self.ledger.add(model, "batch", u["input_tokens"], cached, u["output_tokens"], usd, tag)
        self.cache.put(o["custom_id"], text, {"model": model, "usd": usd})
