# Market-Regime-Analysis-using-Hidden-Markov-Models
A quantitative study using Hidden Markov Models to detect market regimes in the S&amp;P 500 and evaluate regime-based trading strategies. Combines econometric modeling with adaptive decision rules to explore how volatility states influence risk and return dynamics.

Author: Tanvi Madhaw
Institution: University of Groningen – Econometrics & Operations Research

📄 Overview

This project investigates regime‐switching dynamics in financial markets using a Hidden Markov Model (HMM) applied to the S&P 500 index.
The model identifies periods of relative market stability and turbulence based solely on return behavior.
In addition, the project tests regime-dependent trading strategies, both rule-based (fixed probability thresholds) and adaptive (reinforcement learning).
The goal is to demonstrate how probabilistic models can capture nonlinear features of asset returns and how regime awareness can improve decision-making in quantitative investment frameworks.

⚙️ Methodology

1. Data
Daily S&P 500 closing prices (2010–2024) are downloaded via Yahoo Finance and converted to log returns.

3. Model Estimation
Gaussian Hidden Markov Models are estimated for 2–4 regimes.
Model selection is based on the Bayesian Information Criterion (BIC).
The chosen model produces posterior regime probabilities 
P_t(Bull)
P_t(Bear)
.
4. Regime Interpretation
   
 Each inferred regime is characterized by its mean return, volatility, and persistence.
Transition probabilities describe the likelihood of remaining in or switching between regimes.

4. Trading Strategies
Threshold Rules: Go long when 
P_t(Bull) > θ; remain in cash otherwise.

Adaptive Rule: A tabular Q-learning algorithm learns an optimal decision boundary between long and cash positions using regime probabilities as states.

5. Evaluation
Performance is measured via CAGR, Sharpe ratio, and volatility, benchmarked against a buy-and-hold strategy.

🧠 Key Findings

The S&P 500 can be statistically partitioned into two dominant regimes:
Bull: higher mean returns, lower volatility, persistent state.
Bear: negative or low returns, high volatility, short duration.
Regime transitions align closely with major macro-financial events such as the COVID-19 crash (2020) and rate-hike cycle (2022).
Regime-based strategies outperform unconditional buy-and-hold on a risk-adjusted basis.
The Q-learning approach yields a smoother allocation policy than static thresholds, capturing nonlinear boundaries between optimism and caution.

