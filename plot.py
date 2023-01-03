        
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess_df
import numpy as np
import argparse
from preprocess import *

def plot_stock_cluster(df, sample=20, t0=1000, T=100, path="plots/stock_clusters.svg"):
    """Plot a heatmap of the correlation matrix of a random sample of stocks."""
    corr = df.sample(sample, axis=1)[t0:t0 + T].corr().fillna(0)
    sns.color_palette("viridis", as_cmap=True)
    fig = sns.clustermap(1-corr, method='average', figsize=(10,10))
    fig.savefig(path, format='svg', dpi=1200)


def merge_results():
    """Merge the results of the three models into a single dataframe, apply smoothing, and add dates."""
    res = pd.read_csv("results/q1_average.csv")
    res2 = pd.read_csv("results/q1_rtm.csv")
    res3 = pd.read_csv("results/q1_empirical.csv")
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
    """Plot the time series of the effective number of stocks for the three models."""
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    sns.lineplot(data=df, x="date",y="n_effs_rolling", hue="model", ax=ax)
    ax.set(xlabel="Date", ylabel="N_eff", title="Effective number of stocks evolution")


def plot_out_risk(df):
    """Plot the time series of the out-of-sample risk for the three models."""
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    sns.lineplot(data=df, x="date", y="out_risk_rolling", hue="model", ax=ax)
    ax.set(xlabel="Date", ylabel="Risk", title="Out-of-sample risk evolution", ylim=[0,0.05])


def plot_contour(result_dict):
    """
    Plot the contour of the reliability of the portfolios produced by RTM and clustering for different (T, N) pairs.
    """
    perc = lambda x,y: result_dict.get((x,y), np.NaN)
    v_func = np.vectorize(perc)   

    max_T = 1000
    max_N = 500
    Ts = np.linspace(0,max_T, 11)[1:]
    Ns = np.linspace(0, max_N, 11)[1:]
    X,Y= np.meshgrid(Ts, Ns)

    Z = v_func(X,Y).T

    plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    CS = ax.contourf(X,Y,Z, vmin=0, vmax=1)
    ax.set(xlabel="T", ylabel="N")
    fig.colorbar(CS)
    plt.show()


def plot_clippings(result_dict):
    """
    Plot the distributions of the reliability of the clipping methods on a boxplot on the left,
    and the time series of the clipping methods on the right for MP and 0.3.
    """
    df= pd.DataFrame(result_dict)

    df_melted = df.melt(id_vars=None, value_vars=list(df.columns), var_name="model", value_name="value")
    df_melted = df_melted[["model", "value"]]

    fig, ax = plt.subplots(1,2, figsize=(20,8))
    sns.boxplot(data=df_melted, x="model", y="value", ax=ax[0], dodge=False)#ci=99)#
    ax[0].set(ylim=[0,0.6])
    ax[0].legend()

    df[["Marcenko-Pastur", 0.3]].plot(ylim=[0, 0.6], ax=ax[1])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, default=0, help="Question number to execute")

    args = parser.parse_args()

    if args.question == 1:
        df = merge_results()
        plot_effective_number_of_stocks(df)
        plot_out_risk(df)
    if args.question == 2:
        res = load_pickle("results/q2.pkl")
        plot_clippings(res)
    if args.question == 3:
        res = load_pickle("results/q3.pkl")
        print(res)
        plot_contour(res)
     

    plt.show()