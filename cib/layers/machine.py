"""Layer M: probability that a message is machine-generated.

Two detectors, chosen in exploration (T15):
- `ZeroShotLM`: per-message log-likelihood statistics under a small local causal LM (no generator knowledge),
  mapped to a probability by a logistic calibration fitted on exploratory data;
- `CharNgramLR`: logistic regression on character n-grams trained on exploratory synthetic vs organic messages.
Both are fitted on generator A only; transfer is checked on an exploration-only generator and, after the protocol
is frozen, on generator B and fox8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cib.text import clean_for_similarity


class CharNgramLR:
    def __init__(self, seed: int = 0):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000),
            LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", random_state=seed))

    def fit(self, texts: list[str], y: np.ndarray) -> "CharNgramLR":
        self.model.fit([clean_for_similarity(t) for t in texts], y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba([clean_for_similarity(t) for t in texts])[:, 1]


class ZeroShotLM:
    """Mean token log-prob, mean log-rank and mean entropy under a small LM, then logistic calibration."""

    def __init__(self, lm_name: str = "ai-forever/rugpt3small_based_on_gpt2", max_tokens: int = 128,
                 batch_size: int = 16):
        self.lm_name, self.max_tokens, self.batch_size = lm_name, max_tokens, batch_size
        self._lm = self._tok = None
        self.calib = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))

    def _load(self):
        if self._lm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(4)
            self._tok = AutoTokenizer.from_pretrained(self.lm_name)
            self._tok.pad_token = self._tok.pad_token or self._tok.eos_token
            self._lm = AutoModelForCausalLM.from_pretrained(self.lm_name).eval()

    def features(self, texts: list[str]) -> np.ndarray:
        import torch
        self._load()
        out = []
        for s in range(0, len(texts), self.batch_size):
            batch = [clean_for_similarity(t) or "." for t in texts[s:s + self.batch_size]]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)
            with torch.inference_mode():
                logits = self._lm(**enc).logits[:, :-1].float()
            tgt = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].bool()
            logp = torch.log_softmax(logits, -1)
            tok_lp = logp.gather(-1, tgt[..., None])[..., 0]
            rank = (logp > tok_lp[..., None]).sum(-1).float() + 1
            ent = -(logp.exp() * logp).sum(-1)
            n = mask.sum(1).clamp(min=1)
            f = torch.stack([(tok_lp * mask).sum(1) / n, (torch.log(rank) * mask).sum(1) / n,
                             (ent * mask).sum(1) / n, torch.log(n.float())], 1)
            out.append(f.numpy())
        return np.vstack(out)

    def fit(self, texts: list[str], y: np.ndarray) -> "ZeroShotLM":
        self.calib.fit(self.features(texts), y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.calib.predict_proba(self.features(texts))[:, 1]


def account_mean(msgs: pd.DataFrame, prob: pd.Series) -> pd.Series:
    """Mean machine probability per account over its scored messages."""
    return prob.reindex(msgs.msg_id).set_axis(msgs.account_id).groupby(level=0).mean()
