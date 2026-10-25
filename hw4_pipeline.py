"""
Stock Markets Analytics Zoomcamp 2026 - Module 4 Homework
Faithful, headless reproduction of [2026]_Module_04_Colab_Trading_Simulations.ipynb
(hyper-parameter tuning and the DNN section are skipped, exactly as in the notebook,
because they do not feed any prediction used in the simulation).

Usage:
    python hw4_pipeline.py --data stocks_df_combined_2026_09_18.parquet.brotli --scenario base|q1|q2|q3
Scenarios:
    base : notebook "as is"
    q1   : drop 'Ticker' and 'month_wom' from CATEGORICAL before pd.get_dummies (208 features)
    q2   : q1 + Random-Forest tpr/fpr table + extra rules pred12..pred20 (thresholds 0.56..0.64)
    q3   : original notebook, target redefined: is_positive_growth_30d_future = growth_future_30d > 1.022
"""
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--scenario", default="base", choices=["base", "q1", "q2", "q3"])
ap.add_argument("--out", default=None)
args = ap.parse_args()
SC = args.scenario
SHORT_DUMMIES = SC in ("q1", "q2")
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- data + features
df_full = pd.read_parquet(args.data)

if SC == "q3":
    # Q3: redefine the growth target (≈50% positives instead of ≈60%)
    df_full["is_positive_growth_30d_future"] = (df_full["growth_future_30d"] > 1.022).astype(int)

GROWTH = [g for g in df_full.keys() if (g.find("growth_") == 0) & (g.find("future") < 0)]
OHLCV = ["Open", "High", "Low", "Close_x", "Volume"]
CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type"]
TO_PREDICT = [g for g in df_full.keys() if (g.find("future") >= 0)]
TO_DROP = ["Year", "Date", "index_x", "index_y", "index", "Quarter", "Close_y", "Dividends", "Stock Splits"] + CATEGORICAL + OHLCV
df_full["ln_volume"] = df_full.Volume.apply(lambda x: np.log(x + 1e-6))
df_full["div_payout"] = (df_full.Dividends > 0).astype(int)
df_full["stock_split"] = (df_full["Stock Splits"] > 0).astype(int)
CUSTOM_NUMERICAL = ["SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative", "volatility", "ln_volume", "div_payout", "stock_split"]
TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc', 'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix', 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk', 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
 'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase', 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find("cdl") >= 0]
MACRO = ["gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom", "FEDFUNDS", "DGS1", "DGS5", "DGS10"]
NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

df = df_full[df_full.Date >= "2000-01-01"].copy()
df["Month"] = pd.to_datetime(df["Month"], format="%m").dt.strftime("%B")
df["Weekday"] = df["Weekday"].astype("string")
df["wom"] = df.Date.apply(lambda d: (d.day - 1) // 7 + 1)
df["wom"] = df["wom"].astype(str)
df["month_wom"] = df.Month + "_w" + df.wom
del df["wom"]
CATEGORICAL.append("month_wom")

if SHORT_DUMMIES:
    # Q1: fewer dummy variables -> remove Ticker and month_wom right before get_dummies
    CATEGORICAL = [c for c in CATEGORICAL if c not in ("Ticker", "month_wom")]
log("CATEGORICAL used for dummies:", CATEGORICAL)

dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
DUMMIES = dummy_variables.keys().to_list()
df_with_dummies = pd.concat([df, dummy_variables], axis=1)
N_FEATURES = len(NUMERICAL + DUMMIES)
log(f"df_with_dummies[NUMERICAL+DUMMIES] -> {N_FEATURES} features ({len(DUMMIES)} dummies)")

def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
    df["split"] = np.where(df["Date"] <= train_end, "train", np.where(df["Date"] <= val_end, "validation", "test"))
    return df

df_with_dummies = temporal_split(df_with_dummies, df_with_dummies.Date.min(), df_with_dummies.Date.max())
new_df = df_with_dummies.copy()
target_share = new_df["is_positive_growth_30d_future"].mean()
log(f"share of class 1 in target: {target_share:.3f}; growth_future_30d median={new_df.growth_future_30d.median():.4f}")

# ---------------------------------------------------------------- manual predictions
new_df["pred0_manual_cci"] = (new_df.cci > 200).astype(int)
new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)
new_df["pred2_manual_prev_g1_and_snp"] = ((new_df["growth_30d"] > 1) & (new_df["growth_snp500_30d"] > 1)).astype(int)
new_df["pred3_manual_dgs10_5"] = ((new_df["DGS10"] <= 4.5) & (new_df["DGS5"] <= 4)).astype(int)
new_df["pred4_manual_dgs10_fedfunds"] = ((new_df["DGS10"] > 4) & (new_df["FEDFUNDS"] <= 4.795)).astype(int)

def clean(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def fit_decision_tree(X, y, max_depth=20):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf, X.columns

features_list = NUMERICAL + DUMMIES
to_predict = "is_positive_growth_30d_future"
train_valid_df = new_df[new_df.split.isin(["train", "validation"])].copy(deep=True)
valid_df = new_df[new_df.split.isin(["validation"])].copy(deep=True)
test_df = new_df[new_df.split.isin(["test"])].copy(deep=True)
X_train_valid = clean(train_valid_df[features_list + [to_predict]])
X_valid = clean(valid_df[features_list + [to_predict]])
X_test = clean(test_df[features_list + [to_predict]])
X_all = clean(new_df[features_list + [to_predict]].copy(deep=True))
y_train_valid = X_train_valid.pop(to_predict)
y_valid = X_valid.pop(to_predict)
y_test = X_test.pop(to_predict)
y_all = X_all.pop(to_predict)

# ---------------------------------------------------------------- models
log("fit DT depth 10"); clf_10, _ = fit_decision_tree(X_train_valid, y_train_valid, 10)
new_df["pred5_clf_10"] = clf_10.predict(X_all)
log("fit DT depth 4"); clf_4, _ = fit_decision_tree(X_train_valid, y_train_valid, 4)
new_df["pred6_clf_4"] = clf_4.predict(X_all)
log("fit DT depth 7 (clf_best)"); clf_best, _ = fit_decision_tree(X_train_valid, y_train_valid, 7)
new_df["pred7_clf_second_best"] = clf_best.predict(X_all)
log("fit RF depth 13 x100 (rf_best)")
rf_best = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42, n_jobs=-1).fit(X_train_valid, y_train_valid)
log("RF fitted")

# Logistic-regression section of the notebook scales X_test in place (kept for fidelity of tpr table)
scaler = StandardScaler()
X_train_valid_s = scaler.fit_transform(X_train_valid)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------- tpr/fpr table
def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
    scores = []
    thresholds = np.linspace(0, 1, 51) if only_even else np.linspace(0, 1, 101)
    precision = recall = f1_score = 0
    for t in thresholds:
        ap_, an_ = (y_true == 1), (y_true == 0)
        pp, pn = (y_pred >= t), (y_pred < t)
        tp = (pp & ap_).sum(); tn = (pn & an_).sum(); fp = (pp & an_).sum(); fn = (pn & ap_).sum()
        if tp + fp > 0: precision = tp / (tp + fp)
        if tp + fn > 0: recall = tp / (tp + fn)
        if precision + recall > 0: f1_score = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))
    d = pd.DataFrame(scores, columns=["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "accuracy", "f1_score"])
    d["tpr"] = d.tp / (d.tp + d.fn); d["fpr"] = d.fp / (d.fp + d.tn)
    return d

results = {"scenario": SC, "n_features": N_FEATURES, "n_dummies": len(DUMMIES), "target_class1_share": float(target_share)}

if SC == "q2":
    # Comment out clf_best ... / Uncomment rf_best (as instructed). Two variants are reported:
    #  (a) notebook-faithful: X_test after StandardScaler (what the notebook literally does at that cell)
    #  (b) unscaled X_test (what the RF was trained on) -> the meaningful table
    for tag, Xt in [("notebook_scaled_Xtest", X_test_s), ("unscaled_Xtest", X_test.values)]:
        y_pred_test_class1 = rf_best.predict_proba(Xt)[:, 1]
        df_scores = tpr_fpr_dataframe(y_test.values, y_pred_test_class1, only_even=False)
        sub = df_scores[(df_scores.threshold >= 0.5) & (df_scores.threshold <= 0.70)].copy()
        sub["total_trades_tp_fp"] = sub.tp + sub.fp
        print(f"\n===== RF tpr/fpr dataframe ({tag}) thresholds 0.50..0.70 =====")
        print(sub.round(4).to_string(index=False))
        results[f"df_scores_rf_{tag}"] = sub.round(5).to_dict(orient="records")

# ---------------------------------------------------------------- threshold rules
y_pred_all_class1_array = clf_best.predict_proba(X_all)[:, 1]
new_df["proba_pred8"] = y_pred_all_class1_array
new_df["pred8_clf_second_best_rule_56"] = (y_pred_all_class1_array >= 0.56).astype(int)
new_df["proba_pred9"] = y_pred_all_class1_array
new_df["pred9_clf_second_best_rule_66"] = (y_pred_all_class1_array >= 0.66).astype(int)

y_pred_all_class1_array = rf_best.predict_proba(X_all)[:, 1]
new_df["proba_pred10"] = y_pred_all_class1_array
new_df["pred10_rf_best_rule_55"] = (y_pred_all_class1_array >= 0.55).astype(int)
new_df["proba_pred11"] = y_pred_all_class1_array
new_df["pred11_rf_best_rule_65"] = (y_pred_all_class1_array >= 0.65).astype(int)

if SC == "q2":
    for k, thr in enumerate(range(56, 65), start=12):  # pred12_rf_best_rule_56 ... pred20_rf_best_rule_64
        new_df[f"pred{k}_rf_best_rule_{thr}"] = (y_pred_all_class1_array >= thr / 100).astype(int)

# ---------------------------------------------------------------- simulation
PREDICTIONS = [k for k in new_df.keys() if k.startswith("pred")]
sim1_results = []
for pred in PREDICTIONS:
    flt = (new_df.split == "test") & (new_df[pred] == 1)
    sub = new_df[flt]
    n = len(sub)
    gross = (100 * (sub["growth_future_30d"] - 1)).sum()
    fees = -100 * 0.002 * n
    net = gross + fees
    daily = sub.groupby("Date")[pred].count()
    avg_day = daily.mean() if n else np.nan
    q75_day = daily.quantile(0.75) if n else np.nan
    capital = 100 * 30 * q75_day
    cagr = ((capital + net) / capital) ** (1 / 4) if n else np.nan
    sim1_results.append((pred, n, gross, fees, net, (-fees / gross if gross > 0 else None), (net / n if n else None), avg_day, capital, cagr))

cols = ["prediction", "sim1_count_investments", "sim1_gross_rev", "sim1_fees", "sim1_net_rev", "sim1_fees_percentage",
        "sim1_average_net_revenue", "sim1_avg_investments_per_day", "sim1_capital", "sim1_CAGR"]
df_sim1_results = pd.DataFrame(sim1_results, columns=cols)
df_sim1_results["sim1_growth_capital_4y"] = (df_sim1_results.sim1_net_rev + df_sim1_results.sim1_capital) / df_sim1_results.sim1_capital
print("\n===== df_sim1_results =====")
print(df_sim1_results.round(4).to_string())
results["df_sim1_results"] = df_sim1_results.round(6).to_dict(orient="records")

out = args.out or f"results_{SC}.json"
json.dump(results, open(out, "w"), indent=1, default=float)
log("saved", out)
