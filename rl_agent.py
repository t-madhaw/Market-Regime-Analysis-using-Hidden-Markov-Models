"""
Tabular Q-learning agent trained on HMM regime beliefs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def q_learning_train(data, bull_state, n_bins=10, alpha=0.1, gamma=0.99, eps=0.1, fee=0.0005):
    p_bull = data[f"P_state{bull_state}"].values
    rets = data["Returns"].values
    bins = np.linspace(0, 1, n_bins+1)
    states = np.digitize(p_bull, bins) - 1
    Q = np.zeros((n_bins, 2))
    pos = 0
    for t in range(len(rets)-1):
        s = states[t]
        a = np.argmax(Q[s]) if np.random.rand() > eps else np.random.randint(2)
        r = rets[t] if a == 1 else 0
        if a != pos:
            r -= fee
        pos = a
        s_next = states[t+1]
        Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
    return Q, bins

def q_learning_eval(data, Q, bins, bull_state, fee=0.0005):
    p_bull = data[f"P_state{bull_state}"].values
    rets = data["Returns"].values
    states = np.digitize(p_bull, bins) - 1
    pos = 0
    curve = [1.0]
    for t in range(len(rets)):
        s = states[t]
        a = np.argmax(Q[s])
        r = rets[t] if a == 1 else 0
        if a != pos:
            r -= fee
        pos = a
        curve.append(curve[-1]*np.exp(r))
    return np.array(curve[1:])
