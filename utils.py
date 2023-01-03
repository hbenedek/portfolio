from collections import defaultdict
import numpy as np 
from tqdm import tqdm
import itertools
import time
from model import *



def get_markovitz_weights(R: pd.DataFrame, sigma_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the Markowitz weights for the given data and predicted covariance matrix.
    
    Args:
        R (pd.DataFrame): A Pandas dataframe containing the data, with shape (T, N)
        sigma_pred np.ndarray: A NumPy array representing the predicted covariance matrix.
    
    Returns:
        A NumPy array containing the Markowitz weights.
    """

    inv_sigma = np.linalg.pinv(sigma_pred)
    ones = np.ones(R.shape[1])
    w = (ones @ inv_sigma) / (ones @ inv_sigma @ ones)
    return w

def empirical_correlation(df: pd.DataFrame):
    """
    Calculate the empirical correlation for the given dataframe.
    
    Args:
        df (pd.DataFrame): A Pandas dataframe containing the data, with shape (T, N)
    
    Returns:
        A Pandas dataframe containing the empirical correlation.
    """
    return df.corr().fillna(0)


def get_nb_effective_stocks(w: np.ndarray) -> float:
    """    
    This function calculates the number of effective stocks for the given weights by computing
    the reciprocal of the squared L2 norm of the weights.
    
    Args:
        w (np.ndarray): A NumPy array containing the weights.
    
    Returns:
        An integer representing the number of effective stocks.
    """
    return 1 / np.linalg.norm(w)**2

def get_reliability(R, sigma_pred, w) -> float:
    
    sigma = R.corr().fillna(0).values
    realized_risk = w.T @ sigma @ w
    predicted_risk = w.T @ sigma_pred @ w
    return np.abs(realized_risk - predicted_risk) / predicted_risk

def get_risk(R, w) -> float:
    """
    Calculate the risk for the given data and weights.
    
    Args:
        R (pd.DataFrame): A Pandas dataframe containing the data, with shape (T, N)
        w (np.ndarray): A NumPy array containing the weights.
    
    Returns:
        A float representing the risk.
    """
    sigma = R.corr().fillna(0).values
    return w.T @ sigma @ w


def rolling_evolution(df, t, T, correlation_estimate, output_path="results/cluster_result.csv"):
    """
    Calculate the rolling evolution for the given dataframe.
    
    This function calculates the rolling evolution for a given dataframe by dividing the data into
    two parts: an in-sample part and an out-of-sample part. It then applies the given correlation
    estimation method to the in-sample data to compute the predicted covariance matrix, and uses
    this predicted covariance matrix to compute the weights, effective number of stocks, reliability,
    and out-of-sample risk for each time step. The function writes the results of these computations
    to the specified output file in CSV format.
    
    Args:
        df (pd.DataFrame): A Pandas dataframe containing the data to be used for calculating the rolling evolution.
        t (int): An integer representing the starting time step.
        T (int): An integer representing the length of the in-sample and out-of-sample data.
        correlation_estimate: A function that takes in a dataframe and returns a predicted covariance
            matrix. This function will be used to estimate the covariance matrix of the in-sample data.
        output_path (optional): A string representing the path to the output file. The default value
            is "results/cluster_result.csv".
    
    Returns:
        0, indicating that the function completed successfully.
    """
 
    f = open(output_path,'a')
    f.write("t0,n_effs,reliability,out_risk,runtime\n") 
    f.close()

    for t0 in tqdm(range(t,len(df) - T)):
        try:
            R_in, R_out = df.iloc[t0: t0 + int(T/2)], df.iloc[int(T/2): T]

            R_in /= R_in.std()
            R_out /= R_out.std()
            
            start = time.time()
            sigma_pred = correlation_estimate(R_in)
            end = time.time()
        
            w = get_markovitz_weights(R_in, sigma_pred)

            n_eff = get_nb_effective_stocks(w)
            reliability = get_reliability(R_in, sigma_pred, w)
            out_risk = get_risk(R_out, w)
            runtime = end - start

            line = str((t0, n_eff, reliability, out_risk, runtime)).translate({ord('('): '', ord(')'): ''})
            f = open(output_path,'a')
            f.write(f"{line}\n") 
            f.close()
        except:
            print("ERROR OCCURED AT %s", t0)
            line = (str(t0)+",,,,").translate({ord('('): '', ord(')'): ''})
            f = open(output_path,'a')
            f.write(f"{line}\n") 
            f.close()

    return 0


def bootstrapped_reliabilities(df: pd.DataFrame, max_T: int=1000, max_N: int=500, Nboot: int=50, nb_cells: int=101):
    """
    Calculate the bootstrapped reliabilities for the given dataframe.
    
    This function calculates the bootstrapped reliabilities for a given dataframe by performing
    random sampling with replacement and computing the reliabilities using two different methods:
    the RTM method and the average linkage clustering method. The function then compares the 
    reliabilities computed using these two methods for each sample and returns a dictionary that
    maps pairs of values of N and T to the percentage of times that the RTM method produced 
    higher reliabilities than the average linkage clustering method.
    
    Args:
        df (pd.DataFrame): A Pandas dataframe containing the data to be used for calculating the reliabilities.
        max_T (int): An integer representing the maximum length of time window to use in the calculations.
            Defaults to 1000.
        max_N (int): An integer representing the maximum number of stocks to use in the calculations.
            Defaults to 500.
        Nboot (int): An integer representing the number of bootstrap samples to use.
            Defaults to 50.
        nb_cells (optional): An integer representing the number of cells to use when discretizing
            the values of N and T. Defaults to 101.
    
    Returns:
        A dictionary that maps pairs of N and T to the percentage of times that the RTM method
        produced higher reliabilities than the average linkage clustering method.
    """

    Ts = np.linspace(0,max_T, nb_cells)[1:]
    Ns = np.linspace(0, max_N, nb_cells)[1:]
    results = list(itertools.product(Ts, Ns))
    filtered_pairs = [result for result in results if result[0] > (result[1] + 50)]
    nb_dates, nb_assets = df.shape

    NT_to_percentage = {}
    for (N, T) in tqdm(filtered_pairs):
        rtm_results = []
        cluster_results = []
        N, T = int(N), int(T)
        for _ in range(Nboot):
            try:
                t0 = int(np.random.randint(0, nb_dates - T))
                R = df.sample(N, axis=1).iloc[t0: t0 + T]
                R /= R.std()
                
                rtm_corr = rtm_clipped(R)
                cluster_corr = average_linkage_clustering(R)

                w_rtm = get_markovitz_weights(R, rtm_corr)
                w_cluster = get_markovitz_weights(R, cluster_corr)

                reliability_rtm = get_reliability(R, rtm_corr, w_rtm)
                reliability_cluster = get_reliability(R, cluster_corr, w_cluster)

                rtm_results.append(reliability_rtm)
                cluster_results.append(reliability_cluster)
            except:
                pass
        
        percentage = sum([rtm > cluster for (rtm, cluster) in zip(rtm_results, cluster_results)]) / Nboot
        NT_to_percentage[(N, T)] = percentage

    return NT_to_percentage


def bootstrapped_clipping(df: pd.DataFrame, nb_cells: int, N: int, T: int, t: int=0):
    """
    Samples N stocks from the given dataframe and computes the reliabilities using the RTM method in a rolling window.
    for different values of alpha in the range [0, 1] plus the Marcenko-Pastur edge case.

    Args:
        df (pd.DataFrame): A Pandas dataframe containing the data to be used for calculating the reliabilities.
        nb_cells (int): An integer representing the number of cells to use when discretizing
            the values of alpha.
        N (int): An integer representing the number of stocks to use in the calculations.
        T (int): An integer representing the length of time window to use in the calculations.
        t (int): An integer representing the starting index of the time window to use in the calculations.
            Defaults to 0.

    Returns:
        A dictionary that maps the values of alpha to the reliabilities computed using the RTM method.
    """
    df = df.sample(N, axis=1)
    nb_dates, nb_assets = df.shape
    result_dict = defaultdict(list)
    alphas = [i/nb_cells for i in range(1,nb_cells)]
    for t0 in tqdm(range(t,len(df) - T)):
        
        R = df.iloc[t0: t0 + T]
        R /= R.std()
        #R = R.fillna(0)

        rtm_corrs =  rtm_clipped(R, alpha=alphas)
        for rtm_corr, alpha in zip(rtm_corrs, ["Marcenko-Pastur"]+alphas):
            try:
                w_rtm = get_markovitz_weights(R, rtm_corr)
                reliability_rtm = get_reliability(R, rtm_corr, w_rtm)

                result_dict[alpha].append(reliability_rtm)
            except:
                result_dict[alpha].append(np.NaN)                
    return result_dict

