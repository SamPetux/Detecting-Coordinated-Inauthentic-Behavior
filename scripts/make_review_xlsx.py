"""T08: workbook for the author — realism of synthetic comments and a blind human-vs-LLM test.

Contains real post/comment texts, so it is written to data/raw/labels/ (gitignored). The answer key for the blind
sheet is stored separately and never shown in the workbook.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

YELLOW = PatternFill("solid", fgColor="FFF2CC")
OUT = Path("data/raw/labels")


def sheet(wb, title, df, yellow_cols, widths):
    ws = wb.create_sheet(title)
    ws.append(list(df.columns))
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in df.itertuples(index=False):
        ws.append(list(row))
    for j, col in enumerate(df.columns, start=1):
        letter = ws.cell(1, j).column_letter
        ws.column_dimensions[letter].width = widths.get(col, 14)
        for i in range(1, len(df) + 2):
            ws.cell(i, j).alignment = Alignment(wrap_text=True, vertical="top")
            if col in yellow_cols and i > 1:
                ws.cell(i, j).fill = YELLOW
    ws.freeze_panes = "A2"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    syn = pd.read_parquet("data/synthetic/pilot/messages.parquet")
    bg = pd.read_parquet("data/interim/vk_messages.parquet")
    bg = bg[(bg.split == "explore") & ~bg.author_is_community]
    post_text = bg[bg.kind == "post"].set_index("msg_id").text

    camp = syn[syn.campaign_id.notna()].sample(40, random_state=0)
    realism = pd.DataFrame({
        "id": [f"R{i:02d}" for i in range(len(camp))],
        "post_text": camp.target_id.map(post_text).str.slice(0, 600).to_numpy(),
        "comment": camp.text.to_numpy(),
        "realism_author": "", "comment_author": "",
    })

    l3 = syn[syn.level == "L3"].sample(30, random_state=1)
    org = bg[(bg.kind != "post") & bg.text.str.len().between(20, 400)].sample(30, random_state=2)
    blind = pd.concat([pd.DataFrame({"comment": l3.text, "truth": "llm"}),
                       pd.DataFrame({"comment": org.text, "truth": "human"})]).sample(frac=1, random_state=3)
    blind["id"] = [f"B{i:02d}" for i in range(len(blind))]
    key = blind[["id", "truth"]]
    blind = blind.assign(is_ai_author="", comment_author="")[["id", "comment", "is_ai_author", "comment_author"]]

    wb = Workbook()
    ws = wb.active
    ws.title = "инструкция"
    for line in [
        "Проверка синтетических комментариев (исследование координации LLM-ботов). Заполняйте только жёлтые колонки.",
        "",
        "Лист «realism»: 40 комментариев, сгенерированных языковой моделью к реальным постам городских сообществ.",
        "  realism_author — насколько комментарий похож на написанный живым жителем: 1 = явно бот / нелепо,",
        "  3 = сойдёт, но есть странности, 5 = неотличимо от живого. comment_author — что выдаёт (по желанию).",
        "",
        "Лист «blind»: 60 комментариев вперемешку — 30 написаны людьми, 30 языковой моделью. Метки скрыты.",
        "  is_ai_author — «да» (написал ИИ), «нет» (человек) или «не знаю». Не ищите тексты в интернете.",
        "  Цель — узнать, может ли внимательный читатель отличить такие тексты (для статьи: отличает ли человек).",
        "",
        "Тексты постов и живых комментариев — реальные: файл не пересылайте дальше и не публикуйте.",
        "Время: ~40–60 минут. Верните файл в чат.",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    sheet(wb, "realism", realism, {"realism_author", "comment_author"},
          {"post_text": 70, "comment": 60, "comment_author": 30})
    sheet(wb, "blind", blind, {"is_ai_author", "comment_author"}, {"comment": 80, "comment_author": 30})
    wb.save(OUT / "review_synth_v1.xlsx")
    key.to_csv(OUT / "review_synth_v1_key.csv", index=False)
    print(f"{OUT / 'review_synth_v1.xlsx'}: realism {len(realism)}, blind {len(blind)}")


if __name__ == "__main__":
    main()
