from collections import defaultdict
from typing import Dict, Tuple
import pandas as pd
from utils import bootstrapped_reliabilities, rolling_evolution, bootstrapped_clipping, empirical_correlation
from model import *
from preprocess import *
import argparse
from utils import *
  

START = datetime.datetime(2010,1,1)
END = datetime.datetime(2020,12,31)

def download_data(start: datetime.datetime, end: datetime.datetime, path: str="data/nasdaq_raw_2010_2020.csv") -> pd.DataFrame:
    """
    Downloads stock data for the NASDAQ stock exchange within a given time interval and saves it to a CSV file.
    
    Args:
        start (datetime.datetime): The start of the time interval to download data for.
        end (datetime.datetime): The end of the time interval to download data for.
        path (str, optional): The path to the file where the data should be saved. Defaults to "data/nasdaq_raw_2010_2020.csv".
    
    Returns:
        The resulting pandas dataframe.
    """
    tickers = scrape_tickers()
    tickers = tickers
    df = scrape_yahoo(tickers, start, end)
    
    if not os.path.exists("data"):
        os.mkdir("data")
    df.to_csv(path)
    return df


def question_1(T: int, correlation_estimate: callable, output_path: str, path: str="data/nasdaq_raw_2010_2020.csv") -> None:
    """
    Wrapper function for Question 1:
    How does the out-of-sample risk and Neff evolve for different covariance estimates?
    Incrementally performs portfolio optimization, then saves the out-of-sample risk, number of effective stocks, reliability to a csv file.
    For more detail, see the docstring of rolling_evolution()

    Args:
        T (int): Length of the rolling window
        correlation_estimate (callable): The method to use. Must be one of "empirical_correlation", "rtm_clipping", "average_linkage_cluster", "linear_shrinkage", "rie".
        path (str): The path to the file where the data should be loaded from. Defaults to "data/nasdaq_raw_2010_2020.csv".

    Returns:
        None
    """
    df = pd.read_csv(path)
    df = preprocess_df(df)
    print("number of days:", df.shape[0], "number of stocks:", df.shape[1])
    
    f = open(output_path,'a')
    f.write("t0,neffs,reliability,out_risk,runtime\n") 
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
            print("ERROR OCCURED AT", t0)
            line = (str(t0)+",,,,").translate({ord('('): '', ord(')'): ''})
            f = open(output_path,'a')
            f.write(f"{line}\n") 
            f.close()


def question_2(nb_cells: int=10, N:int=200, T:int=300, path: str="data/nasdaq_raw_2010_2020.csv"): 
    """
    Wrapper function for Question 2:
    How does the reliability change in RTM for different cut-off values?
    For N number of randomly selected stocks with a rolling window of T, the RTM method was performed for 
    different cut-off values. The reliability was calculated for each cut-off value and the results were pickled.
    """
    df = pd.read_csv(path)
    df = preprocess_df(df)
    df = df.sample(N, axis=1)
    print("number of days:", df.shape[0], "number of stocks:", df.shape[1])
    
    nb_dates, nb_assets = df.shape
    result_dict = defaultdict(list)
    alphas = [i/nb_cells for i in range(1,nb_cells)]
    for t0 in tqdm(range(0,len(df) - T)):
        
        R = df.iloc[t0: t0 + T]
        R /= R.std()
        #R = R.fillna(0)

        rtm_corrs = rtm_clipped(R, alpha=alphas)
        for rtm_corr, alpha in zip(rtm_corrs, ["Marcenko-Pastur"] + alphas):
            try:
                w_rtm = get_markovitz_weights(R, rtm_corr)
                reliability_rtm = get_reliability(R, rtm_corr, w_rtm)

                result_dict[alpha].append(reliability_rtm)
            except:
                result_dict[alpha].append(np.NaN)    
    return result_dict


def question_3(max_T:int =1000, max_N: int=500, Nboot: int=50, nb_cells: int=101, path: str="data/nasdaq_raw_2010_2020.csv"):
    """
    Wrapper function for Question 3.
    How does the reliability of the models changes in the function of T and N ?
    For each (N, T) pairs performs a Nboot number of bootstrapped portfolio optimization 
    and compare the reliability for the methods "rtm_clipping" and "average_linkage_cluster".

    Args:
        max_T (int): Maximum length of the rolling window.
        max_N (int): Maximum number of stocks in the portfolio.
        Nboot (int): Number of times the calculations should be performed for a given instance.
        path (str): The path to the file where the data should be loaded from. Defaults to "data/nasdaq_raw_2010_2020.csv".

    Returns:
        The resulting dictionary containing for each (N, T) pair the percentage
    """
    df = pd.read_csv(path) 
    df = preprocess_df(df)
    print("number of days:", df.shape[0], "number of stocks:", df.shape[1])

    Ts = np.linspace(0,max_T, nb_cells)[1:]
    Ns = np.linspace(0, max_N, nb_cells)[1:]

    results = list(itertools.product(Ts, Ns))
    nb_dates, nb_assets = df.shape

    NT_to_percentage = {}
    for (N, T) in tqdm(results):
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
                print("ERROR OCCURED AT", (N, T))
        
        percentage = sum([rtm > cluster for (rtm, cluster) in zip(rtm_results, cluster_results)]) / Nboot
        NT_to_percentage[(N, T)] = percentage
    return NT_to_percentage


def question_4(N: int=200, T: int=300, nb_cells: int=11, path: str="data/nasdaq_raw_2010_2020.csv"):
    """
    Wrapper function for Question 4.

    Args:

    Returns:
        
    """
    df = pd.read_csv(path) 
    df = preprocess_df(df)
    df = df.sample(N, axis=1)

    alphas = np.linspace(0, 1, nb_cells)[1:]
    
    results = defaultdict(list)
    for t0 in tqdm(range(0,len(df) - T)):
        R_in, R_out = df.iloc[t0: t0 + int(T/2)], df.iloc[int(T/2): T]
        R_in /= R_in.std()
        R_out /= R_out.std()

        R_in = R_in.corr().fillna(0).values
        R_out = R_out.corr().fillna(0).values

        N = R_in.shape[0]
        I = np.eye(N)
        for alpha in alphas:
            sigma_pred = alpha * R_in + (1 - alpha) * I
            w = get_markovitz_weights(R_in, sigma_pred)
            out_risk = w.T @ R_out @ w
            results[alpha].append(out_risk)
    return results



def question_5(N: int = 200, T: int=300, path: str="data/nasdaq_raw_2010_2020.csv"):
    df = pd.read_csv(path) 
    df = preprocess_df(df)
    df = df.sample(N, axis=1)

    ks = list(np.linspace(0, 50, 11))
    ks = [int(k) for k in ks]
    ks[0] = 1
    results = defaultdict(list)
    for t0 in tqdm(range(0,len(df) - T)):
        R_in, R_out = df.iloc[t0: t0 + int(T/2)], df.iloc[int(T/2): T]
        R_in /= R_in.std()
        R_out /= R_out.std()

        R_in = R_in.corr().fillna(0).values
        R_out = R_out.corr().fillna(0).values

        Fs = filterCovariance(R_in.T, K=ks, Nboot=50, is_correlation=True, method='near')
        for F, k in zip(Fs, ks):
            w = get_markovitz_weights(R_in, F)
            out_risk = w.T @ R_out @ w
            results[k].append(out_risk)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, default=0, help="Question number to execute")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date of the data to download")
    parser.add_argument("--end", type=str, default="2020-12-31", help="End date of the data to download")
    parser.add_argument("--path", type=str, default="data/nasdaq_raw_2010_2020.csv", help="Path to the data file")
    parser.add_argument("--T", type=int, default=300, help="Length of the rolling window for question 1 and 2")
    parser.add_argument("--method", type=str, default="rtm_clipping", help="Method to use for question 1 (rtm, average, empirical)")
    parser.add_argument("--nb_cells", type=int, default=10, help="Number of cells for question 2 or 3")
    parser.add_argument("--N", type=int, default=200, help="Number of stocks for question 2")
    parser.add_argument("--max_T", type=int, default=1000, help="Maximum length of the rolling window for question 3")
    parser.add_argument("--max_N", type=int, default=500, help="Maximum number of stocks in the portfolio for question 3")
    parser.add_argument("--Nboot", type=int, default=50, help="Number of bootstraps for question 3")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for question 1 (linear shrinkage)")
    args = parser.parse_args()
      
    if args.question == 0:
        download_data(args.start, args.end, args.path)
    elif args.question == 1:
        if args.method == "rtm":
            f = rtm_clipped
            output_path = "results/q1_rtm.csv"
        elif args.method == "average":
            f = average_linkage_clustering
            output_path = "results/q1_average.csv"
        elif args.method == "empirical":
            f = empirical_correlation
            output_path = "results/q1_empirical.csv"
        elif args.method == "linear":
            f = lambda R: linear_shrinkage(R, args.alpha)
            output_path = "results/q1_linear.csv"
        elif args.method == "bahc":
            f = bootsrap_halc
            output_path = "results/q1_bahc.csv"
        elif args.method == "rie":
            f = rie_wrapper
            output_path = "results/q1_rie.csv"
        else:
            raise ValueError("Method must be one of rtm, average, empirical, linear, bahc, rie")
        question_1(args.T, f, output_path, args.path)  
    elif args.question == 2:
        result = question_2(args.nb_cells, args.N, args.T, args.path)
        dump_pickle(result, "results/q2.pkl")
    elif args.question == 3:
        result = question_3(args.max_T, args.max_N, args.Nboot, args.nb_cells, args.path)
        dump_pickle(result, "results/q3.pkl")
    elif args.question == 4:
        result = question_4(args.N, args.T, args.nb_cells, args.path)
        dump_pickle(result, "results/q4.pkl")
    elif args.question == 5:
        result = question_5(args.N, args.T, args.path)
        dump_pickle(result, "results/q5.pkl")
    else:
        raise ValueError("Question number must be in [0, 1, 2, 3, 4, 5]")


   
    
    





