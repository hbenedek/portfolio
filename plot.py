        
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess_df
import numpy as np

def plot_stock_cluster(df, sample=20, t0=1000, T=100, path="plots/stock_clusters.svg"):
    corr = df.sample(sample, axis=1)[t0:t0 + T].corr().fillna(0)
    sns.color_palette("viridis", as_cmap=True)
    fig = sns.clustermap(1-corr, method='average', figsize=(10,10))
    fig.savefig(path, format='svg', dpi=1200)


def merge_results():
    res = pd.read_csv("results/empirical_result.csv")
    res2 = pd.read_csv("results/rtm_result.csv")
    res3 = pd.read_csv("results/cluster_result.csv")
    res["model"] = "Empirical"
    res2["model"] = "RMT"
    res3["model"] = "Clustering"
    res.bfill(inplace=True)
    res2.bfill(inplace=True)
    res3.bfill(inplace=True)

    res["n_effs_rolling"] = res["neffs"].rolling(10, min_periods=1).mean()
    res2["n_effs_rolling"] = res2["neffs"].rolling(10, min_periods=1).mean()
    res3["n_effs_rolling"] = res3["neffs"].rolling(10, min_periods=1).mean()

    res["out_risk_rolling"] = res["out_risk"].rolling(10, min_periods=1).mean()
    res2["out_risk_rolling"] = res2["out_risk"].rolling(10, min_periods=1).mean()
    res3["out_risk_rolling"] = res3["out_risk"].rolling(10, min_periods=1).mean()

    df = pd.read_csv("data/nasdaq_raw_2010_2020.csv")
    df = preprocess_df(df)
    
    res["date"] = df.iloc[:-252].index
    res["date"] = pd.to_datetime(res["date"])
    res2["date"] = pd.to_datetime(res["date"])
    res3["date"] = pd.to_datetime(res["date"])
    
    df = pd.concat([res, res2, res3])

    df.index = np.arange(7560)

    return df

def plot_effective_number_of_stocks(df):
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    sns.lineplot(data=df, x="date",y="n_effs_rolling", hue="model", ax=ax)
    ax.set(xlabel="Date", ylabel="N_eff", title="Effective number of stocks evolution")


def plot_out_risk(df):
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    sns.lineplot(data=df, x="date", y="out_risk_rolling", hue="model", ax=ax)
    ax.set(xlabel="Date", ylabel="Risk", title="Out-of-sample risk evolution", ylim=[0,0.05])

