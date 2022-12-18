from typing import Dict, Tuple
import pandas as pd
from utils import bootstrapped_reliabilities, rolling_evolution, bootstrapped_clipping
from model import *
from preprocess import *
  

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
    df = scrape_yahoo(tickers, start, end)
    df.to_csv(path)
    return df


def question_1(T: int, method: callable, path: str="data/nasdaq_raw_2010_2020.csv") -> None:
    """
    Wrapper function for Question 1:
    How does the out-of-sample risk and Neff evolve for different covariance estimates?
    Incrementally performs porfolio optimization, then saves the out-of-sample risk, number of effective stocks, reliability to a csv file.
    For more detail, see the docstring of rolling_evolution()

    Args:
        T (int): Lenght of the rolling window
        method (callable): The method to use. Must be one of "empirical_correlation", "rtm_clipping", "average_linkage_clsuter".
        path (str): The path to the file where the data should be loaded from. Defaults to "data/nasdaq_raw_2010_2020.csv".

    Returns:
        None
    """
    df = pd.read_csv(path)
    df = preprocess_df(df)
    rolling_evolution(df, 0, T, method)


def question_2(nb_cells: int=10, N:int=200, T:int=300, path: str="data/nasdaq_raw_2010_2020.csv"):
    df = pd.read_csv(path)
    df = preprocess_df(df)
    result = bootstrapped_clipping(df, nb_cells, N, T, t=0)
    print(result)
    dump_pickle(result, "q2.pkl")



def question_3(max_T:int =1000, max_N: int=500, Nboot: int=50, nb_cells: int=101, path: str="data/nasdaq_raw_2010_2020.csv"):
    """
    Wrapper function for Question 3.
    How does the reliability of the models changes in the function of T and N ?
    For each (N, T) pairs performs a Nboot numbe rof bootstrapped portfolio optimization 
    and compare the reliability for the methods "rtm_clipping" and "average_linkage_cluster".

    Args:
        max_T (int): Maximum lenght of the rolling window.
        max_N (int): Maximum number of stocks in the portfolio.
        Nboot (int): Number of times the calculations should be performed for a given instance.
        path (str): The path to the file where the data should be loaded from. Defaults to "data/nasdaq_raw_2010_2020.csv".

    Returns:
        The resulting dictionary containing for each (N, T) pair the percentage
    """
    df = pd.read_csv(path)
    df = preprocess_df(df)
    rel_dict = bootstrapped_reliabilities(df, max_T=max_T, max_N=max_N, Nboot=Nboot, nb_cells=nb_cells)
    dump_pickle(rel_dict, "NT.pkl")
    return rel_dict

if __name__ == "__main__":
    question_2()