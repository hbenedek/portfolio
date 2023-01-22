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
    return abs(realized_risk - predicted_risk) / predicted_risk

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

