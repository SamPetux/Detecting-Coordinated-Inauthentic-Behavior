"""Persona sampler for synthetic accounts."""
from __future__ import annotations

import random

AGES = ["18–25", "26–35", "36–50", "51–65", "65+"]
GENDERS = ["мужчина", "женщина"]
DISTRICTS = ["Автозаводский", "Канавинский", "Ленинский", "Московский", "Нижегородский", "Приокский", "Советский",
             "Сормовский"]
LENGTHS = ["очень коротко, 3–8 слов", "коротко, одно предложение", "два-три предложения"]
REGISTERS = ["разговорный, можно без заглавных букв и с пропущенными запятыми", "нейтральный, грамотный",
             "эмоциональный, с восклицаниями"]
MANNERS = ["задаёт риторический вопрос", "делится личным опытом", "жалуется", "иронизирует", "коротко соглашается",
           "спорит с кем-то в комментариях"]
TYPOS = [0.0, 0.02, 0.05]
EMOJI = ["без эмодзи", "иногда один эмодзи в конце"]


def sample(rng: random.Random) -> dict:
    return {
        "age": rng.choice(AGES), "gender": rng.choice(GENDERS), "district": rng.choice(DISTRICTS),
        "length": rng.choice(LENGTHS), "register": rng.choice(REGISTERS), "manner": rng.choice(MANNERS),
        "typo_rate": rng.choice(TYPOS), "emoji": rng.choice(EMOJI),
    }


def describe(p: dict) -> str:
    return (f"{p['gender']}, {p['age']} лет, живёт в районе: {p['district']}. Пишет {p['length']}; стиль — "
            f"{p['register']}; обычно {p['manner']}; {p['emoji']}.")


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    """Swap two adjacent letters in a word with probability `rate` per eligible word."""
    if rate <= 0:
        return text
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            j = rng.randrange(1, len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)
