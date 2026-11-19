"""Confirmatory analysis strictly by docs/06_protocol.md. Reads results/confirm/*.csv only.

Writes docs/07_confirmatory_results.md (aggregates) and docs/results/confirm_*.csv.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from cib.stats import holm

ALPHA = 0.001
BASELINES = ["raw_T", "raw_S", "copypaste", "luceri", "account_machine"]


def pair_key(campaign: str) -> str:
    """confirm-<brief>-<level>-s<seed*1000+i>-<tag> -> <brief>|<seed number>: the same campaign across levels."""
    m = re.match(r"confirm-(.+)-L\d-s(\d+)-", campaign)
    return f"{m.group(1)}|{m.group(2)}"


def one_sided(diff: np.ndarray) -> float:
    """H1: median(diff) > 0; zero differences split (zsplit)."""
    d = np.asarray(diff, float)
    if np.allclose(d, 0) or len(d) < 2:
        return 1.0
    return float(wilcoxon(d, zero_method="zsplit", alternative="greater").pvalue)


def campaign_table(pc: pd.DataFrame, fpr: float = ALPHA) -> pd.DataFrame:
    """Recall per (level, pair key, detector), pooled over generators of the campaign's accounts."""
    x = pc[pc.fpr == fpr].copy()
    x["hit"] = x.recall * x.n
    g = x.groupby(["level", "campaign", "detector"])[["hit", "n"]].sum()
    g["recall"] = g.hit / g.n
    g = g.reset_index()
    g["key"] = g.campaign.map(pair_key)
    return g.pivot_table(index=["level", "key"], columns="detector", values="recall")


def main() -> None:
    summ = pd.concat([pd.read_csv(f) for f in glob.glob("results/confirm/L*_s*.csv")], ignore_index=True)
    pc = pd.concat([pd.read_csv(f) for f in glob.glob("results/confirm/percampaign_*.csv")], ignore_index=True)
    ct = campaign_table(pc)
    out, p_formal = [], {}

    # H1a: copypaste drop L0->L3 >= 0.30 and M* drop <= 0.10; test: copypaste drop > M* drop (paired)
    l0, l3 = ct.loc["L0"], ct.loc["L3"]
    keys = l0.index.intersection(l3.index)
    drop_cp = (l0.loc[keys, "copypaste"] - l3.loc[keys, "copypaste"])
    drop_ms = (l0.loc[keys, "M_star"] - l3.loc[keys, "M_star"])
    p_formal["H1a"] = one_sided((drop_cp - drop_ms).to_numpy())
    h1a = drop_cp.mean() >= 0.30 and drop_ms.mean() <= 0.10
    out.append(("H1a", f"падение копипасты L0→L3 = {drop_cp.mean():.3f} (≥ 0,30); падение M* = {drop_ms.mean():.3f} "
                       f"(≤ 0,10); campaigns = {len(keys)}", h1a))

    # H1b: generator-B accounts at L3: machine detector <= M* - 0.30
    b = pc[(pc.level == "L3") & (pc.fpr == ALPHA) & (pc.group == "gpt-5-mini")]
    bt = b.pivot_table(index="campaign", columns="detector", values="recall")
    diff = bt.M_star - bt.account_machine - 0.30
    p_formal["H1b"] = one_sided(diff.to_numpy())
    h1b = (bt.account_machine.mean() <= bt.M_star.mean() - 0.30)
    out.append(("H1b", f"генератор B, L3: M* = {bt.M_star.mean():.3f}, детектор машинного текста = "
                       f"{bt.account_machine.mean():.3f} (критерий: ≤ M* − 0,30); campaigns = {len(bt)}", h1b))

    # H2: L4-L5: M* >= max(layer T, layer S) + 0.05 and >= best comparison + 0.05
    hi = ct.loc[["L4", "L5"]]
    best_layer = hi[["layer_T", "layer_S"]].max(axis=1)
    best_cmp = max(BASELINES, key=lambda c: hi[c].mean())
    p1 = one_sided((hi.M_star - best_layer - 0.05).to_numpy())
    p2 = one_sided((hi.M_star - hi[best_cmp] - 0.05).to_numpy())
    p_formal["H2"] = max(p1, p2)
    h2 = (hi.M_star - best_layer).mean() >= 0.05 and (hi.M_star - hi[best_cmp]).mean() >= 0.05
    out.append(("H2", f"L4–L5: M* = {hi.M_star.mean():.3f}; лучший слой (T/S) = {best_layer.mean():.3f}; лучший метод "
                      f"сравнения = {best_cmp} {hi[best_cmp].mean():.3f}; campaigns = {len(hi)}", h2))

    # H4: M* >= M0 + 0.03 on L4-L5
    p_formal["H4"] = one_sided((hi.M_star - hi.M0 - 0.03).to_numpy())
    h4 = (hi.M_star - hi.M0).mean() >= 0.03
    out.append(("H4", f"L4–L5: M* = {hi.M_star.mean():.3f}, M0 = {hi.M0.mean():.3f}", h4))

    adj = holm(p_formal)

    # H3: burst ratio at 1% (M0 vs raw T), bootstrap over runs
    s = summ.set_index("detector")
    r_m0 = summ[summ.detector == "full"]["h3_ratio@0.01"].replace(np.inf, np.nan).dropna()
    r_rt = summ[summ.detector == "raw_T"]["h3_ratio@0.01"].replace(np.inf, np.nan).dropna()
    rng = np.random.default_rng(0)
    boot = [rng.choice(r_rt.to_numpy(), len(r_rt)).mean() - rng.choice(r_m0.to_numpy(), len(r_m0)).mean()
            for _ in range(2000)]
    lo = float(np.quantile(boot, 0.025))
    h3 = r_m0.mean() <= 1.0 and (r_rt.mean() - r_m0.mean()) >= 0.5
    out.append(("H3", f"M0 = {r_m0.mean():.3f}; raw T = {r_rt.mean():.3f}; разность = {r_rt.mean() - r_m0.mean():.3f} "
                      f"(95 % нижняя граница {lo:.3f})", h3))

    # H5: benign flag at 1% for M*, binomial on the distinct benign accounts (same controls in every run)
    ms = summ[summ.detector == "M_star"]
    n_benign = int(ms.n_benign.iloc[0])
    rate = float(ms["benign_flag@0.01"].mean())
    k = int(round(rate * n_benign))
    p5 = binomtest(k, n_benign, 0.02, alternative="less").pvalue
    am = float(summ[summ.detector == "account_machine"]["benign_flag@0.01"].mean())
    h5 = rate <= 0.02
    out.append(("H5", f"M*: {rate:.4f} ({k} из {n_benign}; биномиальный против 2 %: p = {p5:.3g}); детектор машинного "
                      f"текста: {am:.3f}", h5))

    lines = ["# Результаты подтверждающей проверки", "",
             "Строго по `docs/06_protocol.md` (заморожен 2026-11-18). Recall — доля аккаунтов кампании при пороге 0,1 % "
             "ложных тревог на органике; единица — кампания; тесты — односторонний Уилкоксон, поправка Холма.", "",
             "Гипотеза подтверждена, только если выполнены и точечный критерий, и тест (H1a, H1b, H2, H4: p Холма < 0,05; "
             "H5: биномиальный p < 0,05; H3 — по бутстрэп-границе).", "",
             "| Гипотеза | Значения | Точечный критерий | p | Итог по протоколу |", "|---|---|---|---|---|"]
    test_ok = {h: adj[h] < 0.05 for h in adj}
    test_ok["H5"] = p5 < 0.05
    test_ok["H3"] = lo >= 0.5
    p_show = dict(adj, H5=p5)
    for h, txt, ok in out:
        p = p_show.get(h, float("nan"))
        verdict = "подтверждена" if ok and test_ok.get(h, False) else "не подтверждена"
        lines.append(f"| {h} | {txt} | {'да' if ok else 'нет'} | {p:.3g} | {verdict} |")
    lv = summ.groupby(["detector", "level"])["recall@0.001"].mean().unstack("level")
    lines += ["", "## Recall при FPR 0,1 % по уровням (среднее по seed 11–13)", "", lv.round(3).to_markdown(), ""]
    Path("docs/07_confirmatory_results.md").write_text("\n".join(lines) + "\n")
    lv.to_csv("docs/results/confirm_recall_by_level.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
