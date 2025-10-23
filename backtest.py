"""
Utility functions for evaluating strategies on HMM-based signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def regime_strategy(data: pd.DataFrame, bull_state: int, thresh: float = 0.6,
                    fee: float = 0.0005) -> np.ndarray:
    """
    Simulates a simple strategy: go long when P(Bull) >= threshold, else cash.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ['Returns', f'P_state{bull_state}'].
    bull_state : int
        Index of the Bull regime from HMM labeling.
    thresh : float
        Probability cutoff for entering a long position.
    fee : float
        One-way transaction cost (subtracted when switching positions).

    Returns
    -------
    np.ndarray
        Equity curve array representing growth of $1.
    """
    p_bull = data[f"P_state{bull_state}"].values
    rets = data["Returns"].values

    position = 0  # 0 = cash, 1 = long
    curve = [1.0]

    for t in range(len(rets)):
        desired = 1 if p_bull[t] >= thresh else 0
        r = rets[t] if desired == 1 else 0.0
        if desired != position:  # switching -> pay transaction fee
            r -= fee
        position = desired
        curve.append(curve[-1] * np.exp(r))

    return np.array(curve[1:])


def evaluate_performance(curve: np.ndarray, returns: np.ndarray,
                         freq: int = 252) -> dict:
    """
    Computes CAGR and Sharpe ratio for a strategy.

    Parameters
    ----------
    curve : np.ndarray
        Strategy equity curve (growth of $1).
    returns : np.ndarray
        Daily log returns used in the strategy.
    freq : int
        Trading days per year (default 252).

    Returns
    -------
    dict
        CAGR, Sharpe ratio, and annual volatility.
    """
    T = len(curve)
    cagr = curve[-1] ** (freq / T) - 1
    daily = np.diff(np.log(curve))
    mu, sigma = daily.mean() * freq, daily.std() * np.sqrt(freq)
    sharpe = mu / sigma if sigma > 0 else 0
    vol = sigma
    return {"CAGR": cagr, "Sharpe": sharpe, "Volatility": vol}


def plot_equity_curves(data: pd.DataFrame, curves: dict,
                       title: str = "Equity Curves", save_path: str = None):
    """
    Plots multiple equity curves on the same chart.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'Date' column.
    curves : dict
        {label: np.ndarray} pairs of equity curves.
    title : str
        Plot title.
    save_path : str
        Optional path to save figure.
    """
    plt.figure(figsize=(10, 5))
    for label, c in curves.items():
        plt.plot(data["Date"], c, label=label)
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
