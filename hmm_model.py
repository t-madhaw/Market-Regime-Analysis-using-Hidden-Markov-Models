"""
Fit Gaussian Hidden Markov Models and label market regimes.
"""
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from numpy.linalg import eig

def fit_hmm(returns, n_components=2, random_state=42):
    model = GaussianHMM(
        n_components=n_components, covariance_type="full",
        n_iter=2000, random_state=random_state
    )
    model.fit(returns.reshape(-1, 1))
    return model

def select_model(returns, k_list=(2,3,4)):
    results = []
    best_model = None
    for k in k_list:
        model = fit_hmm(returns, n_components=k)
        logL = model.score(returns)
        n_params = k + k + k*(k-1)
        bic = -2 * logL + n_params * np.log(len(returns))
        results.append({"k": k, "logL": logL, "BIC": bic})
        if not best_model or bic < best_model["BIC"]:
            best_model = {"model": model, "k": k, "BIC": bic}
    return best_model["model"], pd.DataFrame(results)

def label_regimes(data, model):
    X = data["Returns"].values.reshape(-1, 1)
    data["Regime"] = model.predict(X)
    posterior = model.predict_proba(X)
    for s in range(model.n_components):
        data[f"P_state{s}"] = posterior[:, s]
    # Label by volatility (Bear) and mean (Bull)
    stats = data.groupby("Regime")["Returns"].agg(["mean", "std"])
    bear = stats["std"].idxmax()
    bull = stats["mean"].idxmax()
    label_map = {bear: "Bear", bull: "Bull"}
    data["RegimeLabel"] = data["Regime"].map(label_map)
    return data, label_map, stats, model.transmat_
