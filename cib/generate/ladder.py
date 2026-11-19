"""Attacker ladder L0..L5 and the benign-LLM-user control.

Two phases: `plan_*` builds a deterministic table of actions (account, target, time, prompt); `realize` turns it
into message rows once texts exist. Planning never calls the API, so one plan can run online (pilot) or via batch.

Levels: L0 identical text; L1 one template per target filled per account; L2 one LLM call per (account, target)
without persona; L3 = L2 + persona and a generator per account; L4 = L3 texts with a delay/participation grid;
L5 = L4 with dense camouflage history. Every level gets a light account history (neutral comments around the
campaign), otherwise campaign-only accounts are trivially separable by activity span.
"""
from __future__ import annotations

import random
import re
import uuid

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from cib import schema
from cib.generate import personas, timing
from cib.generate.briefs import Brief

SYSTEM = """Ты помогаешь исследователям создать синтетический набор данных для проверки детекторов скоординированных
кампаний в социальных сетях. Тексты никогда не публикуются на реальных платформах и используются только офлайн.

Задача: написать ОДИН комментарий так, как его написал бы обычный житель Нижнего Новгорода под постом в городском
сообществе ВКонтакте. Требования к тексту:
- русский язык, живая разговорная речь, как в реальных комментариях в соцсетях;
- без хэштегов, без ссылок, без упоминания, что текст написан искусственным интеллектом или по заданию;
- не повторяй формулировки поста дословно и не пересказывай его целиком;
- не начинай с обращений вроде «Конечно!», «Вот комментарий:», не ставь текст в кавычки;
- не используй канцелярит и «правильные» обороты пресс-релизов, если стиль автора этого не предполагает;
- не упоминай реальных политиков, партии, национальности и религии; не призывай к насилию и нарушению закона;
- если задана позиция, выражай её естественно, как личное мнение; не обязательно прямо, можно через пример из жизни;
- если задан комментарий, на который надо ответить, отвечай именно ему, коротко, как в ветке обсуждения;
- если задан образ автора, соблюдай его длину, стиль, манеру и пол (согласование глаголов прошедшего времени).

Ответ — только текст комментария, одна строка, без пояснений.

Справка о стиле городских сообществ: жители пишут о дворах, транспорте, ЖКХ, ремонте дорог, ценах, работе
поликлиник и школ; часто обращаются к администрации, иронизируют, делятся бытовыми историями, спорят друг с другом,
используют сокращения («щас», «ну», «вообще», «реально»), иногда пишут без заглавной буквы и без точки в конце.
Длинные комментарии редки; типичный комментарий — от трёх слов до двух предложений. Восклицательные знаки и
многоточия встречаются, смайлики — реже. Встречаются вопросы к другим комментаторам («а вы сами там были?»),
оценки («наконец-то», «как всегда»), ссылки на личный опыт («у нас во дворе так же»), упоминания районов и улиц.
"""

REFUSAL = re.compile(r"(как (ии|искусственный интеллект|языковая модель)|я не могу|не могу помочь|извините,|"
                     r"^вот (ваш |мой )?комментарий|^конечно[!,]|as an ai|i can't|i cannot)", re.I)


def clean_output(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    t = text.strip().strip('"«»').strip()
    if not t or REFUSAL.search(t) or len(t) > 600:
        return None
    return t.replace("\n", " ")


def select_targets(bg: pd.DataFrame, brief: Brief, k: int, rng: random.Random, *, min_comments: int = 5,
                   pool: int = 400, window_days: tuple[int, ...] = (7, 14, 21),
                   n_communities: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Topically close organic posts with enough comments, inside one 1-3 week window, from 1-3 communities.

    Without the time window and with a forced spread over communities, campaign accounts were trivially separable
    from organic ones by activity span and community count (artifact audit, 2026-11-18).
    """
    posts = bg[(bg.kind == "post") & (bg.text.str.len() > 40)]
    n_com = bg[bg.kind != "post"].groupby("target_id").size()
    posts = posts[posts.msg_id.map(n_com).fillna(0) >= min_comments]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(posts.text.str.lower())
    q = vec.transform([f"{brief.query} {brief.goal}".lower()])
    sim = (X @ q.T).toarray().ravel()
    cand = posts.iloc[np.argsort(-sim)[:pool]]
    span = rng.choice(window_days) * 86400
    n_c = rng.choice(n_communities)
    lo, hi = int(cand.ts.min()), int(cand.ts.max()) - span
    best = None
    for _ in range(200):
        start = rng.randint(lo, max(lo, hi))
        win = cand[(cand.ts >= start) & (cand.ts < start + span)]
        top = win.community.value_counts().index[:n_c]
        win = win[win.community.isin(top)]
        if len(win) >= k:
            return win.sample(n=k, random_state=rng.randrange(2**31)).sort_values("ts")
        if best is None or len(win) > len(best):
            best = win
    # sparse topics: fall back to the fullest window found, padded from the same communities
    extra = cand[cand.community.isin(best.community.unique()) & ~cand.msg_id.isin(best.msg_id)]
    extra = extra.iloc[np.argsort(np.abs(extra.ts.to_numpy() - best.ts.median()))[: k - len(best)]]
    return pd.concat([best, extra]).sort_values("ts")


ORGANIC_REPLY_NOTE = ("Это ответ на чужой комментарий в обсуждении под этим постом (текст комментария не показан): "
                      "напиши короткую реплику в ветке, не цитируя собеседника.")


def _prompt(post_text: str, goal: str | None, persona: dict | None, reply_to: str | None) -> str:
    parts = [f"Пост в сообществе:\n{post_text[:1500]}"]
    if reply_to:
        parts.append(f"Ответь на этот комментарий в ветке:\n{reply_to}")
    if goal:
        parts.append(f"Позиция автора комментария: {goal}.")
    if persona:
        parts.append(f"Образ автора: {personas.describe(persona)}")
    else:
        parts.append("Длина: коротко, как обычный комментарий, не больше 25 слов.")
    return "\n\n".join(parts)


def new_accounts(n: int, salt: str, rng: random.Random, scope: str = "") -> list[str]:
    """`scope` (campaign id) keeps ids distinct when several levels share one seed and thus the same targets."""
    return [schema.hash_account(f"syn:{scope}:{uuid.UUID(int=rng.getrandbits(128))}", salt) for _ in range(n)]


def plan_campaign(bg: pd.DataFrame, brief: Brief, *, level: str, n_accounts: int, n_targets: int, salt: str,
                  seed: int, generators: list[str], split: str, median_delay_s: float = 1200,
                  participation: float = 1.0, p_reply: float = 0.2, p_reply_organic: float = 0.4,
                  p_followup: float = 0.35, history_lambda: float = 4.0) -> pd.DataFrame:
    """One row per planned campaign action; `req_*` columns describe the text request (shared rows share a key).

    Besides the first message per (account, target), an account may reply to an organic comment in the thread
    (the organic text is never sent to the API) and may come back with a follow-up, as organic users do.
    """
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    cid = f"{split}-{brief.id}-{level}-s{seed}"
    targets = select_targets(bg, brief, n_targets, rng)
    accs = new_accounts(n_accounts, salt, rng, scope=cid)
    pers = [personas.sample(rng) for _ in accs]
    gen = [generators[i % len(generators)] for i in range(n_accounts)] if level in ("L3", "L4", "L5") \
        else [generators[0]] * n_accounts
    mask = timing.participation(nrng, n_accounts, n_targets, participation)
    org = bg[bg.kind != "post"][["msg_id", "target_id", "ts"]]
    org_by_target = {t: g.sort_values("ts") for t, g in org[org.target_id.isin(targets.msg_id)].groupby("target_id")}
    use_persona = level in ("L3", "L4", "L5")
    shared = level in ("L0", "L1")
    rows = []

    def organic_parent(target: str, ts: int) -> tuple[str | None, int, bool]:
        """(parent, time, attempted). A random organic comment of the thread posted at most an hour after the
        planned time; the reply is moved after it. Random draws do not depend on the outcome, so prompts (and the
        text cache) stay identical across planner fixes."""
        g = org_by_target.get(target)
        if g is None or g.empty:
            return None, ts, False
        i = rng.randrange(len(g))
        lag = int(timing.delays(nrng, 1, 300)[0])
        # a parent posted hours later would drag the campaign reply out of its burst
        near = g[g.ts <= ts + 3600]
        if not len(near):
            return None, ts, True
        j = near.index[i % len(near)]
        return g.msg_id.at[j], max(ts, int(g.ts.at[j]) + lag), True

    for t_i, (_, post) in enumerate(targets.iterrows()):
        who = np.flatnonzero(mask[:, t_i])
        ts = post.ts + timing.delays(nrng, len(who), median_delay_s)
        events = [(int(ts[j]), int(who[j]), 0) for j in range(len(who))]
        for t0, a, _ in list(events):
            k, t_prev = 1, t0
            while rng.random() < p_followup and k < 5:
                t_prev += int(timing.delays(nrng, 1, 900)[0])
                events.append((t_prev, a, k))
                k += 1
        events.sort()
        earlier: list[int] = []
        for t_ev, a, follow in events:
            reply_idx, parent_org = None, None
            r = rng.random()
            if earlier and r < p_reply:
                reply_idx = rng.choice(earlier)
            attempted = False
            if not (earlier and r < p_reply) and r < p_reply + p_reply_organic:
                parent_org, t_ev, attempted = organic_parent(post.msg_id, t_ev)
                if attempted and parent_org is None and earlier:
                    # no live thread yet: answer the latest campaign message instead (no extra random draw)
                    reply_idx = earlier[-1]
            prompt = _prompt(post.text, brief.goal, pers[a] if use_persona else None, None)
            if attempted:
                prompt += "\n\n" + ORGANIC_REPLY_NOTE
            key = f"{cid}|{post.msg_id}" if shared else f"{cid}|{post.msg_id}|{a}|{follow}"
            rows.append({
                "campaign_id": cid, "level": level, "split": split, "account_id": accs[a], "account_idx": a,
                "ts": t_ev, "target_id": post.msg_id, "community": post.community, "reply_to_row": reply_idx,
                "parent_org": parent_org, "model": gen[a], "persona": pers[a] if use_persona else None,
                "req_kind": "template" if level == "L1" else "text", "req_key": key, "req_prompt": prompt,
            })
            earlier.append(len(rows) - 1)
    plan = pd.DataFrame(rows)
    plan["row_id"] = [f"{cid}#{i}" for i in range(len(plan))]
    plan["reply_to"] = [plan.row_id.iat[int(r)] if pd.notna(r) else None for r in plan.reply_to_row]
    plan = plan.drop(columns="reply_to_row")
    if history_lambda > 0:
        hist = plan_camouflage(bg, accs, pers, gen, cid, level, split, history_lambda, seed,
                               center_ts=int(targets.ts.median()))
        plan = pd.concat([plan, hist], ignore_index=True)
    return plan


def plan_camouflage(bg: pd.DataFrame, accs: list[str], pers: list[dict], gen: list[str], cid: str, level: str,
                    split: str, lam: float, seed: int, center_ts: int, p_reply_organic: float = 0.5,
                    p_followup_hist: float = 0.4) -> pd.DataFrame:
    """Account history: neutral comments on random organic posts in a 20-80 day span around the campaign.

    Campaign-only accounts were separable from organic ones by activity span alone (artifact audit, 2026-11-18).
    Half of the history messages are replies to organic comments; their texts are never sent to the API.
    """
    rng, nrng = random.Random(seed + 1), np.random.default_rng(seed + 1)
    posts = bg[(bg.kind == "post") & (bg.text.str.len() > 40)]
    org = bg[bg.kind != "post"][["msg_id", "target_id", "ts"]]
    rows = []
    hours_org = ((org.ts // 3600) % 24).value_counts(normalize=True).reindex(range(24), fill_value=0)
    post_hour = ((posts.ts // 3600) % 24).to_numpy()
    for a, acc in enumerate(accs):
        n = 1 + int(nrng.poisson(lam))
        half = rng.uniform(15, 60) * 86400
        h0 = int(nrng.choice(24, p=hours_org.to_numpy()))
        # organic users have a daily routine: weight posts by circular distance to the account's active hour
        dist = np.minimum(np.abs(post_hour - h0), 24 - np.abs(post_hour - h0))
        in_win = ((posts.ts >= center_ts - half) & (posts.ts <= center_ts + half)).to_numpy()
        w = np.exp(-(dist ** 2) / (2 * 2.5 ** 2)) * np.where(in_win, 1.0, 1e-6)
        n_threads = max(1, int(round(n / (1 + p_followup_hist))))
        idx = nrng.choice(len(posts), size=min(n_threads, len(posts)), replace=False, p=w / w.sum())
        picks = []
        for i in idx:
            picks.append(i)
            while len(picks) < n and rng.random() < p_followup_hist:
                picks.append(i)
        for i in picks:
            post = posts.iloc[int(i)]
            ts = int(post.ts + timing.delays(nrng, 1, 1800, 1.2)[0])
            parent = None
            if rng.random() < p_reply_organic:
                cand = org[org.target_id == post.msg_id]
                if len(cand):
                    i = rng.randrange(len(cand))
                    parent = cand.msg_id.iat[i]
                    ts = max(ts, int(cand.ts.iat[i]) + int(timing.delays(nrng, 1, 300)[0]))
            prompt = _prompt(post.text, None, pers[a], None) + ("\n\n" + ORGANIC_REPLY_NOTE if parent else "")
            rows.append({"campaign_id": cid, "level": level, "split": split, "account_id": acc, "account_idx": a,
                         "ts": ts, "target_id": post.msg_id, "community": post.community, "reply_to": None,
                         "parent_org": parent, "model": gen[a], "persona": pers[a], "req_kind": "text",
                         "req_key": f"{cid}|hist|{post.msg_id}|{a}|{len(rows)}", "req_prompt": prompt,
                         "row_id": f"{cid}#hist{len(rows)}"})
    return pd.DataFrame(rows)


def plan_benign(bg: pd.DataFrame, *, n_accounts: int, salt: str, seed: int, generators: list[str], split: str,
                min_msgs: int = 3, max_msgs: int = 15) -> pd.DataFrame:
    """Independent LLM users: own persona, random posts, organic delays, no shared goal (negatives for H5)."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    posts = bg[(bg.kind == "post") & (bg.text.str.len() > 40)]
    accs = new_accounts(n_accounts, salt, rng, scope=f"benign-{split}-s{seed}")
    rows = []
    for a, acc in enumerate(accs):
        p = personas.sample(rng)
        for _, post in posts.sample(n=rng.randint(min_msgs, max_msgs), random_state=rng.randrange(2**31)).iterrows():
            rows.append({"campaign_id": None, "level": "benign_llm", "split": split, "account_id": acc,
                         "account_idx": a, "ts": int(post.ts + timing.delays(nrng, 1, 1800, 1.2)[0]),
                         "target_id": post.msg_id, "community": post.community, "reply_to": None,
                         "model": generators[a % len(generators)], "persona": p, "req_kind": "text",
                         "req_key": f"benign-{split}-s{seed}|{post.msg_id}|{a}",
                         "req_prompt": _prompt(post.text, None, p, None),
                         "row_id": f"benign-{split}-s{seed}#{len(rows)}"})
    return pd.DataFrame(rows)


def plan_benign_hot(bg: pd.DataFrame, *, n_accounts: int, n_posts: int, salt: str, seed: int,
                    generators: list[str], split: str, per_account: tuple[int, int] = (3, 8)) -> pd.DataFrame:
    """Hard negative control for H5: independent LLM users who all comment the same busy threads.

    They co-occur by chance (same posts, organic timing) and share LLM style, but have no common goal; a detector
    that flags them is detecting LLM text, not coordination."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    n_com = bg[bg.kind != "post"].groupby("target_id").size()
    posts = bg[(bg.kind == "post") & (bg.text.str.len() > 40)]
    hot = posts[posts.msg_id.map(n_com).fillna(0) >= n_com.quantile(0.95)].sample(n=n_posts, random_state=seed)
    org = bg[bg.kind != "post"]
    accs = new_accounts(n_accounts, salt, rng, scope=f"benignhot-{split}-s{seed}")
    rows = []
    for a, acc in enumerate(accs):
        p = personas.sample(rng)
        for _, post in hot.sample(n=rng.randint(*per_account), random_state=rng.randrange(2**31)).iterrows():
            # organic timing: copy the delay of a random organic comment of the same thread
            d = org[org.target_id == post.msg_id].ts - post.ts
            ts = int(post.ts + (d.iat[rng.randrange(len(d))] if len(d) else timing.delays(nrng, 1, 1800)[0]))
            rows.append({"campaign_id": None, "level": "benign_hot", "split": split, "account_id": acc,
                         "account_idx": a, "ts": ts + rng.randint(0, 59), "target_id": post.msg_id,
                         "community": post.community, "reply_to": None, "parent_org": None,
                         "model": generators[a % len(generators)], "persona": p, "req_kind": "text",
                         "req_key": f"benignhot-{split}-s{seed}|{post.msg_id}|{a}",
                         "req_prompt": _prompt(post.text, None, p, None),
                         "row_id": f"benignhot-{split}-s{seed}#{len(rows)}"})
    return pd.DataFrame(rows)


TEMPLATE_INSTR = ("Напиши шаблон комментария с вариантами в фигурных скобках через вертикальную черту, например: "
                  "«{Наконец-то|Давно пора}, {сделали|взялись}...». 3–5 мест с вариантами. Только шаблон.")


def fill_template(tpl: str, rng: random.Random) -> str:
    return re.sub(r"\{([^{}]*)\}", lambda m: rng.choice(m.group(1).split("|")).strip(), tpl)


def _needs_parent(plan: pd.DataFrame) -> pd.Series:
    # shared levels copy one text per target, so their replies need no parent-specific request
    return plan.reply_to.notna() & ~plan.level.isin(["L0", "L1"])


def requests(plan: pd.DataFrame, texts: dict[str, str | None] | None = None) -> pd.DataFrame:
    """Unique text requests (req_key, model, instructions, prompt).

    Round 1 (texts=None): everything except in-campaign replies. Round 2: replies, with the parent's generated
    text inserted into the prompt; parents whose text failed are skipped (the reply is dropped later).
    """
    need = _needs_parent(plan)
    if texts is None:
        sel = plan[~need].copy()
    else:
        sel = plan[need].copy()
        parent_key = sel.reply_to.map(plan.set_index("row_id").req_key).to_numpy()
        parent_txt = [clean_output(texts.get(k)) for k in parent_key]
        sel["req_prompt"] = [f"{p}\n\nОтветь на этот комментарий в ветке:\n{t}" if t else None
                             for p, t in zip(sel.req_prompt, parent_txt)]
        sel = sel[sel.req_prompt.notna()]
    req = sel.drop_duplicates("req_key")[["req_key", "model", "req_kind", "req_prompt"]].copy()
    req["instructions"] = np.where(req.req_kind == "template", SYSTEM + "\n" + TEMPLATE_INSTR, SYSTEM)
    return req


def realize(plan: pd.DataFrame, texts: dict[str, str | None], *, seed: int) -> pd.DataFrame:
    """Plan + generated texts -> message rows (schema). Rows whose text failed are dropped."""
    rng = random.Random(seed)
    out = plan.copy()
    raw = [texts.get(f"reply|{r}", texts.get(k)) if isinstance(rt, str) else texts.get(k)
           for r, k, rt in zip(out.row_id, out.req_key, out.reply_to)]
    is_tpl = out.req_kind == "template"
    out["text"] = [fill_template(t, rng) if (tp and isinstance(t, str)) else t for t, tp in zip(raw, is_tpl)]
    out["text"] = [clean_output(t) for t in out.text]
    out["text"] = [personas.add_typos(t, p["typo_rate"], rng) if (isinstance(t, str) and isinstance(p, dict)) else t
                   for t, p in zip(out.text, out.persona)]
    # reply texts are keyed by the reply row, so look them up by row_id before falling back to the shared key
    out["msg_id"] = "syn_" + out.row_id
    porg = out["parent_org"] if "parent_org" in out else pd.Series(None, index=out.index)
    out["parent_id"] = ["syn_" + r if isinstance(r, str) else (o if isinstance(o, str) else None)
                        for r, o in zip(out.reply_to, porg)]
    out["kind"] = np.where(out.parent_id.notna(), "reply", "comment")
    out = out[out.text.notna()]
    out = out.assign(source="synthetic", is_injected=True)
    return schema.conform(out[[c for c in out.columns if c in schema.COLUMNS]])
