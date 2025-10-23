"""
main.py
--------
Market Regime Analysis using Hidden Markov Models (HMM)
and a reinforcement learning extension for regime-based trading.

Author: Tanvi Madhaw
University of Groningen | Econometrics and Operations Research
"""

# ---------- Imports ----------
import os
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.hmm_model import select_model, label_regimes
from src.rl_agent import q_learning_train, q_learning_eval
from src.backtest import regime_strategy, evaluate_performance, plot_equity_curves
from src.visualisations import plot_regimes
from src.utils import save_df

# ---------- Configuration ----------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKER = "^GSPC"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
TRANSACTION_FEE = 0.0005
THRESHOLDS = [0.5, 0.6, 0.7]

# ---------- 1. Load and Prepare Data ----------
print("Loading market data...")
data = load_data(TICKER, START_DATE, END_DATE)
print(f"Dataset: {len(data)} daily observations from {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")

# ---------- 2. Hidden Markov Model Estimation ----------
print("\nEstimating Hidden Markov Models...")
returns = data["Returns"].values.reshape(-1, 1)
model, model_summary = select_model(returns, k_list=(2, 3, 4))

print("\nModel Selection Results (BIC Comparison):")
print(model_summary)

data, label_map, stats, transmat = label_regimes(data, model)
bull_state = [k for k, v in label_map.items() if v == "Bull"][0]

print("\nRegime Characteristics:")
print(stats.assign(label=stats.index.map(label_map)))

print("\nEstimated Transition Matrix:")
print(transmat)

save_df(model_summary, os.path.join(RESULTS_DIR, "bic_comparison.csv"))
save_df(stats.assign(label=stats.index.map(label_map)), os.path.join(RESULTS_DIR, "regime_summary.csv"))
pd.DataFrame(transmat).to_csv(os.path.join(RESULTS_DIR, "transition_matrix.csv"), index=False)

# ---------- 3. Visualize Identified Regimes ----------
print("\nPlotting identified market regimes...")
plot_regimes(data, label_map)

# ---------- 4. Fixed-Threshold Regime Strategies ----------
print("\nEvaluating regime-dependent investment strategies...")
baseline_results, curves = [], {}

for threshold in THRESHOLDS:
    curve = regime_strategy(data, bull_state, thresh=threshold, fee=TRANSACTION_FEE)
    metrics = evaluate_performance(curve, data["Returns"].values)
    metrics["Threshold"] = threshold
    baseline_results.append(metrics)
    curves[f"p(Bull) ≥ {threshold}"] = curve

baseline_df = pd.DataFrame(baseline_results)
save_df(baseline_df, os.path.join(RESULTS_DIR, "baseline_strategies.csv"))

plot_equity_curves(
    data,
    curves,
    title="Performance of Threshold-Based Regime Strategies",
    save_path=os.path.join(RESULTS_DIR, "baseline_equity_curves.png")
)

print("\nSummary of Fixed-Threshold Strategies:")
print(baseline_df.round(4))

# ---------- 5. Learning-Based Regime Strategy ----------
print("\nApplying reinforcement learning to identify adaptive regime policy...")
Q, bins = q_learning_train(data, bull_state=bull_state)
rl_curve = q_learning_eval(data, Q, bins, bull_state=bull_state)

# Buy-and-Hold Benchmark
bh_curve = np.exp(np.cumsum(data["Returns"].values))
curves_rl = {"Buy & Hold": bh_curve, "Regime Policy": rl_curve}

plot_equity_curves(
    data,
    curves_rl,
    title="Regime-Adapted vs Buy & Hold Performance",
    save_path=os.path.join(RESULTS_DIR, "rl_vs_bh.png")
)

# Performance Summary
T = len(data)
cagr_rl = rl_curve[-1] ** (252 / T) - 1
cagr_bh = bh_curve[-1] ** (252 / T) - 1

print("\nPerformance Comparison:")
print(f"Regime-Adaptive Strategy CAGR: {cagr_rl:.4f}")
print(f"Buy & Hold CAGR:               {cagr_bh:.4f}")

# Save Q-table for interpretation
q_table = pd.DataFrame(Q, columns=["Q_cash", "Q_long"])
q_table["bin_left"], q_table["bin_right"] = bins[:-1], bins[1:]
save_df(q_table, os.path.join(RESULTS_DIR, "q_table.csv"))

print("\nAll results exported to the 'results' directory.")
print("Analysis complete.")
