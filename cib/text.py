"""Text cleaning and normalisation for similarity layers."""
from __future__ import annotations

import re

URL = re.compile(r"https?://\S+|www\.\S+")
VK_MENTION = re.compile(r"\[(?:id|club|public)\d+\|([^\]]*)\]")
SPACES = re.compile(r"\s+")


def clean_for_similarity(text: str) -> str:
    """Drop URLs, keep the visible name from VK mentions, collapse whitespace."""
    t = URL.sub(" ", text or "")
    t = VK_MENTION.sub(r"\1", t)
    return SPACES.sub(" ", t).strip()


def sem_eligible(text: str, min_chars: int = 20) -> bool:
    return len(clean_for_similarity(text)) >= min_chars
