import pandas as pd
import requests
import io
import yfinance as yf
import datetime
from tqdm import tqdm
import numpy as np
import fastcluster
from math import ceil
import os
import seaborn as sns
import time
import pickle
from typing import List, Any

def scrape_tickers():
    """
    Scrapes a list of ticker symbols for companies listed on the NASDAQ stock exchange.
    
    Returns:
        List[str]: A list of ticker symbols for companies listed on the NASDAQ stock exchange.
    """
    url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
    s = requests.get(url).content
    companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
    tickers = companies['Symbol'].tolist()
    return tickers


def scrape_yahoo(tickers: List[str], start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """
    Scrapes stock data from Yahoo Finance for the given ticker symbols and time interval.

    Args:
        tickers (List[int]): List containing the tickers of the stocks that should be scraped.
        start (datetime.datetime): The start of the time interval to download data for.
        end (datetime.datetime): The end of the time interval to download data for.

    Returns:
        pd.DataFrame: Pandas dataframe containing the prices of the specified stocks.
    """
    # create empty dataframe
    stock_final = pd.DataFrame()
    # iterate over each symbol
    for ticker in tickers:  
        try:
            stock = yf.download(ticker,start=start, end=end, progress=False)
            # append the individual stock prices 
            if len(stock) == 0:
                None
            else:
                stock['Name']=ticker
                stock_final = stock_final.append(stock,sort=False)
        except Exception:
            None
    return stock_final


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a dataframe of stock data. Keeps only the adjusted closed price, calculates the returns on each stock.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    df = df[["Date","Adj Close", "Name"]]
    df.set_index("Date", inplace=True)
    df = df.pivot(columns="Name", values="Adj Close")
    df = ((df / df.shift(1)) - 1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.clip(lower=1e-3, upper=1e3, inplace=True)
    df.fillna(0, inplace=True)
    return df


def dump_pickle(object_, path: str) -> None:
    """
    Dumps an object to a pickle file at the specified path.

    Args:
        object_: The object to dump.
        path (str): The path to the pickle file where the object should be dumped.

    Returns:
        None
    """
    with open(path, 'wb') as f:
        pickle.dump(object_, f)
        

def load_pickle(path: str) -> Any:
    """
    Loads an object from a pickle file at the specified path.

    Args:
        path (str): The path to the pickle file where the object is stored.

    Returns:
        Any: The loaded object.
    """
    with open(path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object






