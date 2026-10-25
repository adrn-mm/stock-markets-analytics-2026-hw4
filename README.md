# Stock Markets Analytics Zoomcamp 2026 — Module 4 Homework

Headless reproduction of `[2026]_Module_04_Colab_Trading_Simulations.ipynb` (`hw4_pipeline.py`), run on
`stocks_df_combined_2026_09_18.parquet.brotli`. Each question uses a fresh run (`--scenario q1|q2|q3`).

## Answers (from the Google Colab run of `HW4_Module04_Solution.ipynb`)

| Q | Answer |
|---|--------|
| **Q1** — CAGR of `pred10_rf_best_rule_55` with fewer dummies (208 features) | **1.1359** (up from 1.1195) |
| **Q2** — threshold of the second-best RF rule with avg trades/day < 5 | **0.56** (`pred12_rf_best_rule_56`: CAGR 1.1177, 4.40 trades/day) |
| **Q3** — number of trades of `pred5_clf_10` after target = growth_future_30d > 1.022 | **7337** (CAGR 1.1218, 8.01 trades/day) |
| **Q4** | see `Q4_strategy.md` |

### Q1 details
`CATEGORICAL = ['Month','Weekday','ticker_type']` → 22 dummies + 186 numerical = **208 features** (original: 301).
`pred10_rf_best_rule_55`: 3,514 trades, net revenue $11,962, capital $18,000, avg 5.16 trades/day, **CAGR 1.1359**.

### Q2 details (RF thresholds 0.55–0.65, Q1 setup)
| rule | trades | avg/day | CAGR |
|---|---|---|---|
| 0.55 | 3514 | 5.16 | 1.1359 |
| **0.56** | **2641** | **4.40** | **1.1177** |
| 0.57 | 1889 | 3.81 | 1.0912 |
| 0.58 | 1330 | 3.46 | 1.0665 |
| 0.59 | 899 | 3.08 | 1.0613 |
| 0.60 | 583 | 2.78 | 1.0442 |
| 0.61 | 352 | 2.51 | 1.0402 |
| 0.62 | 204 | 2.08 | 1.0259 |
| 0.63 | 123 | 1.68 | 1.0193 |
| 0.64 | 65 | 1.44 | 1.0089 |
| 0.65 | 31 | 1.35 | 1.0057 |

Higher thresholds → fewer trades, but CAGR keeps falling. 0.55 is the best rule overall (slightly above 5 trades/day);
0.56 is the best within the "< 5 trades/day" subspace, with CAGR close to 1.12.

### Q3 details
Class-1 share drops from 59.0% to 49.1%. Best CAGR is `pred5_clf_10` (1.1218) with **7,337 trades** (8.01/day),
vs 9,699 trades (10.6/day) for the same model on the original target.

> Note: Random Forest results depend on the scikit-learn/numpy version even with `random_state=42`; the numbers above
> come from Colab (the course reference environment). Decision-tree results (Q3) are identical across environments.

## Files
- `HW4_Module04_Solution.ipynb` — **main notebook**, executed with outputs; all 4 questions in one place
- `HW4_Q1_Colab.ipynb`, `HW4_Q2_Colab.ipynb`, `HW4_Q3_Colab.ipynb` — fresh copies of the official 2026 Colab notebook with only the homework edits (marked 🔧 HW4), plus an answer cell at the end; run them on Colab as-is
- `hw4_pipeline.py` — pipeline, `python hw4_pipeline.py --data <parquet> --scenario q1|q2|q3`
- `log_q2.txt`, `log_q3.txt`, `results_q2.json`, `results_q3.json` — full outputs (Q2 run also contains the Q1 setup results)
- `Q4_strategy.md` — exploratory answer
