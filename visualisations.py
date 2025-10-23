
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_regimes(data, label_map):
    colors = {"Bull": "tab:blue", "Bear": "tab:orange"}
    plt.figure(figsize=(12,6))
    for s, lbl in label_map.items():
        plt.plot(data.loc[data["Regime"]==s, "Date"], 
                 data.loc[data["Regime"]==s, "Price"],
                 label=lbl, color=colors.get(lbl,"gray"))
    plt.title("Market Regimes via Hidden Markov Model")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()
