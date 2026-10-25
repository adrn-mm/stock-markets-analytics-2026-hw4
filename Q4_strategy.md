# Q4 — My ideal trading strategy

**Strategy: "Top-K high-confidence ensemble with volatility-scaled sizing"**

1. **Signal.** Each day, score every stock with the Random Forest probability (`proba_pred10`, short-dummy setup from Q1) and only consider candidates where the probability is ≥ the efficient threshold found in Q2 *and* the depth-7 Decision Tree also votes 1 (a simple 2-model agreement filter that removes noisy, low-conviction signals).
2. **Selection.** Rank the remaining candidates by probability and take at most the **top 3 per day** — this caps the daily trade count (time efficiency, fewer fees) while keeping the highest-precision bets.
3. **Sizing.** Instead of a flat $100, allocate a fixed daily budget (≈1/30 of capital, since each position is held 30 trading days) split across the selected names **inversely to their recent volatility**, with a hard cap of 10% of capital per ticker and 30% per `ticker_type` (sector/region) to avoid concentration.
4. **Risk management.** Skip new entries when the market regime is weak (e.g., S&P 500 30-day growth < 1 or VIX elevated), use a stop-loss around −8% and a take-profit around +15% within the 30-day holding window, and never let the same ticker exceed one open position at a time.
5. **Implementation / validation.** Implement it as a daily rebalancing loop on the TEST period that tracks cash and open positions explicitly (rather than the "$100 per signal" approximation), include the 0.2% round-trip fees, and compare CAGR, max drawdown, Sharpe ratio and average trades per day against `pred10_rf_best_rule_55`. Thresholds and K are tuned on the VALIDATION period only, to avoid look-ahead bias.
