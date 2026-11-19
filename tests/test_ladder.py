from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from cib import schema
from cib.generate import ladder, runner
from cib.generate.briefs import BRIEFS
from cib.generate.llm import LLM, Price


def background(n_posts=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    words = ["парковка платная центр", "автобус маршрут", "отопление батареи", "парк сквер деревья", "школа дети"]
    for i in range(n_posts):
        pid = f"g{i % 4}_p{i}"
        rows.append(dict(msg_id=pid, account_id="a" * 16, ts=1_700_000_000 + i * 3600, kind="post", target_id=pid,
                         community=f"g{i % 4}", text=f"Новость: {words[i % 5]} обсуждают жители района номер {i}",
                         source="vk", split="explore"))
        for j in range(6):
            rows.append(dict(msg_id=f"{pid}_c{j}", account_id=f"{j:016d}", ts=1_700_000_000 + i * 3600 + 60 * j,
                             kind="comment", target_id=pid, community=f"g{i % 4}", text="ок", source="vk",
                             split="explore"))
    return schema.conform(pd.DataFrame(rows))


class FakeResponses:
    def create(self, **kw):
        self.n = getattr(self, "n", 0) + 1
        txt = "{Да|Ага}, {хорошо|отлично}" if "шаблон" in kw["instructions"] else f"ответ {self.n}"
        return NS(output_text=txt, usage=NS(input_tokens=10, output_tokens=5, input_tokens_details=NS(cached_tokens=0)))


@pytest.fixture
def llm(tmp_path):
    return LLM(NS(responses=FakeResponses()), {"m": Price(1, 0.1, 1)}, cache_path=str(tmp_path / "c.sqlite"),
               ledger_path=str(tmp_path / "l.csv"), budget_usd=10)


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_campaign_levels(level, llm):
    bg = background()
    plan = ladder.plan_campaign(bg, BRIEFS[0], level=level, n_accounts=6, n_targets=4, salt="s", seed=1,
                                generators=["m"], split="explore", p_reply=0.5)
    msgs = runner.generate(plan, llm, seed=1)
    schema.validate(msgs)
    assert len(msgs) == len(plan), "fake LLM never fails, so no row may be dropped"
    hist = msgs.msg_id.str.contains("#hist")
    camp = msgs[~hist]
    assert msgs.account_id.nunique() == 6 and camp.target_id.nunique() == 4
    assert hist.any() and set(msgs[hist].account_id) <= set(camp.account_id)
    assert msgs.is_injected.all() and (msgs.campaign_id.str.contains(level)).all()
    assert (msgs.ts > msgs.target_id.map(bg.set_index("msg_id").ts)).all()
    texts_per_target = camp[camp.kind == "comment"].groupby("target_id").text.nunique()
    if level == "L0":
        assert (texts_per_target == 1).all()
    if level in ("L2", "L3"):
        assert msgs.parent_id.dropna().isin(pd.concat([msgs.msg_id, bg.msg_id])).all()
        # independent calls per account: identical prompts must not collapse into one cached answer
        assert msgs.text.nunique() == len(msgs)


def test_plan_is_deterministic():
    bg = background()
    a = ladder.plan_campaign(bg, BRIEFS[3], level="L3", n_accounts=5, n_targets=3, salt="s", seed=7,
                             generators=["m"], split="explore")
    b = ladder.plan_campaign(bg, BRIEFS[3], level="L3", n_accounts=5, n_targets=3, salt="s", seed=7,
                             generators=["m"], split="explore")
    pd.testing.assert_frame_equal(a, b)


def test_benign_has_no_campaign(llm):
    plan = ladder.plan_benign(background(), n_accounts=3, salt="s", seed=0, generators=["m"], split="explore")
    msgs = runner.generate(plan, llm)
    assert msgs.campaign_id.isna().all() and (msgs.level == "benign_llm").all()
    assert not schema.account_labels(msgs).any()


def test_clean_output_rejects_meta():
    assert ladder.clean_output("Конечно! Вот комментарий: класс") is None
    assert ladder.clean_output("«норм сделали»") == "норм сделали"


def test_targets_inside_window_and_few_communities():
    import random
    from cib.generate.ladder import select_targets
    bg = background(n_posts=400)
    for seed in range(5):
        t = select_targets(bg, BRIEFS[0], 6, random.Random(seed), window_days=(7,), n_communities=(1, 2))
        assert len(t) == 6 and (t.ts.max() - t.ts.min()) <= 7 * 86400
        assert t.community.nunique() <= 2
