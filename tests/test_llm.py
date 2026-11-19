import json
from types import SimpleNamespace as NS

import pytest

from cib.generate.llm import LLM, BudgetExceeded, Price

PRICES = {"m": Price(input=1.0, cached_input=0.1, output=10.0)}


class FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        usage = NS(input_tokens=1000, output_tokens=100, input_tokens_details=NS(cached_tokens=0))
        return NS(output_text=f"reply to {kw['input']}", usage=usage)


def make(tmp_path, budget=1.0):
    client = NS(responses=FakeResponses())
    llm = LLM(client, PRICES, cache_path=str(tmp_path / "c.sqlite"), ledger_path=str(tmp_path / "l.csv"),
              budget_usd=budget)
    return llm, client


def test_cache_hit_does_not_pay_twice(tmp_path):
    llm, client = make(tmp_path)
    a = llm.generate("m", "sys", "hello", est_input_tokens=1000, max_output_tokens=100)
    b = llm.generate("m", "sys", "hello", est_input_tokens=1000, max_output_tokens=100)
    assert a == b == "reply to hello" and client.responses.calls == 1
    assert llm.ledger.total() == pytest.approx((1000 * 1.0 + 100 * 10.0) / 1e6)


def test_budget_stop(tmp_path):
    llm, client = make(tmp_path, budget=0.0001)
    with pytest.raises(BudgetExceeded):
        llm.generate("m", "sys", "hello", est_input_tokens=1000, max_output_tokens=100)
    assert client.responses.calls == 0


def test_batch_cost_is_half_and_cached_tokens_cheaper(tmp_path):
    llm, _ = make(tmp_path)
    assert llm.cost("m", 1000, 0, 100, batch=True) == pytest.approx(llm.cost("m", 1000, 0, 100) / 2)
    assert llm.cost("m", 1000, 1000, 0) < llm.cost("m", 1000, 0, 0)


def test_absorb_batch_line(tmp_path):
    llm, _ = make(tmp_path)
    line = {"custom_id": "k1", "response": {"status_code": 200, "body": {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 0}}}}}
    llm._absorb_batch_line(json.loads(json.dumps(line)), "m", "t")
    assert llm.cache.get("k1") == "ok"
    assert llm.ledger.total() > 0


def test_chunks_respect_size():
    from cib.generate.llm import _chunks
    lines = ["x" * 99] * 10
    parts = _chunks(lines, 450)
    assert sum(len(p) for p in parts) == 10 and all(sum(len(l) + 1 for l in p) <= 450 for p in parts)


def test_failed_batch_raises(tmp_path):
    llm, _ = make(tmp_path)
    b = NS(id="b1", status="failed", errors="too big", output_file_id=None)
    llm.client = NS(files=NS(create=lambda **k: NS(id="f")), batches=NS(create=lambda **k: b, retrieve=lambda i: b))
    with pytest.raises(RuntimeError):
        llm.run_batch("m", [("sys", "hello")], est_input_tokens=10, max_output_tokens=10, poll_s=0)
