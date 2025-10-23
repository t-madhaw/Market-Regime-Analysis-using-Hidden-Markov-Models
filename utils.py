"""
Utility helpers for metrics and saving results.
"""
import numpy as np
import pandas as pd

def sharpe_ratio(returns, freq=252):
    mu, sigma = np.mean(returns)*freq, np.std(returns)*np.sqrt(freq)
    return mu/sigma if sigma>0 else 0

def save_df(df, path):
    df.to_csv(path, index=False)
