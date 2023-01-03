from typing import Dict, Tuple
import pandas as pd
from utils import bootstrapped_reliabilities, rolling_evolution, bootstrapped_clipping, empirical_correlation
from model import *
from preprocess import *
from plot import *
import argparse
  

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
    df.to_csv(path)
    return df


def question_1(T: int, method: callable, output_path: str, path: str="data/nasdaq_raw_2010_2020.csv") -> None:
    """
    Wrapper function for Question 1:
    How does the out-of-sample risk and Neff evolve for different covariance estimates?
    Incrementally performs portfolio optimization, then saves the out-of-sample risk, number of effective stocks, reliability to a csv file.
    For more detail, see the docstring of rolling_evolution()

    Args:
        T (int): Length of the rolling window
        method (callable): The method to use. Must be one of "empirical_correlation", "rtm_clipping", "average_linkage_cluster".
        path (str): The path to the file where the data should be loaded from. Defaults to "data/nasdaq_raw_2010_2020.csv".

    Returns:
        None
    """
    df = pd.read_csv(path)
    df = preprocess_df(df)
    print("number of days:", df.shape[0], "number of stocks:", df.shape[1])
    rolling_evolution(df, 0, T, method, output_path)


def question_2(nb_cells: int=10, N:int=200, T:int=300, path: str="data/nasdaq_raw_2010_2020.csv"): 
    """
    Wrapper function for Question 2:
    How does the reliability change in RTM for different cut-off values?
    For N number of randomly selected stocks with a rolling window of T, the RTM method was performed for 
    different cut-off values. The reliability was calculated for each cut-off value and the results were pickled.
    """
    df = pd.read_csv(path)
    df = preprocess_df(df)

    print("number of days:", df.shape[0], "number of stocks:", df.shape[1])
    result = bootstrapped_clipping(df, nb_cells, N, T, t=0)
    return result


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

    rel_dict = bootstrapped_reliabilities(df, max_T=max_T, max_N=max_N, Nboot=Nboot, nb_cells=nb_cells)
    return rel_dict

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
    args = parser.parse_args()
      
    if args.question == 0:
        download_data(args.start, args.end, args.path)
    if args.question == 1:
        if args.method == "rtm":
            f = rtm_clipped
            output_path = "q1_rtm.csv"
        elif args.method == "average":
            f = average_linkage_clustering
            output_path = "q1_average.csv"
        elif args.method == "empirical":
            f = empirical_correlation
            output_path = "q1_empirical.csv"
        else:
            raise ValueError("Method must be one of rtm, average, empirical")
        question_1(args.T, f, output_path, args.path)  
    elif args.question == 2:
        result = question_2(args.nb_cells, args.N, args.T, args.path)
        dump_pickle(result, "results/q2.pkl")
        plot_clippings(result)
    elif args.question == 3:
        result = question_3(args.max_T, args.max_N, args.Nboot, args.nb_cells, args.path)
        dump_pickle(result, "results/q3.pkl")
        print(result)
        plot_contour(result)

    else:
        raise ValueError("Question number must be in [0, 1, 2, 3]")


   
    
    





