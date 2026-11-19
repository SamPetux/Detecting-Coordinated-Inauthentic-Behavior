# VK data report (aggregates only)

Source: `data/raw/corpus_cib_2026-11.csv.gz` → `data/interim/vk_messages.parquet`. Split boundaries (MSK): explore < 2026-06-01T00:00:00+03:00, confirm < 2026-11-16T00:00:00+03:00; later messages dropped.

| Split | Posts | Comments | Replies | Accounts | Communities |
|---|---|---|---|---|---|
| confirm | 29348 | 103898 | 104564 | 36133 | 44 |
| explore | 16800 | 77025 | 83325 | 30657 | 44 |

## Exploratory split profile

| Statistic | Value |
|---|---|
| Actions per account: median / p90 / p99 / max | 1 / 9 / 66 / 4420 |
| Accounts with 1 action / ≥10 / ≥100 | 15908 / 2854 / 175 |
| Share of replies among comments | 0.520 |
| Community-authored share of actions | 0.145 |
| Delay post → first comment, min: median / p90 | 30 / 1153 |
| Comments per post: median / p90 | 5 / 42 |
| Busiest hour (MSK) share / quietest hour share | 0.064 / 0.009 |
| Texts shorter than 20 chars (comments) | 0.075 |

Confirmatory split: counts only until the protocol is frozen (T16).
