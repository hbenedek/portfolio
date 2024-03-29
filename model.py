import numpy as np
import fastcluster
import pandas as pd
from numbers import Complex, Integral, Real
from math import ceil
from typing import Union, Tuple, List
import rie_estimator


########## LINEAR SHRINKAGE ##########

def linear_shrinkage(R: np.ndarray, alpha: float) -> np.ndarray:
    """
    Performs linear shrinkage on the given correlation matrix `R`.

    Args:
        R (np.ndarray): Correlation matrix of shape (N, N), where `N` is the number of features (e.g. stock tickers).
        alpha (float): Shrinkage parameter.
    
    Returns:
        np.ndarray: Shrinked correlation matrix of shape (N, N).
    """
    R = R.corr().fillna(0).values
    return alpha * R + (1 - alpha) * np.eye(R.shape[0])


########## HALC ##########


def dist(R: np.ndarray):
    """
    Computes the distances between clusters in the average linkage clustering dendrogram of the given correlation matrix `R`.
    
    
    Args:
        R (np.ndarray): Correlation matrix of shape (N, N), where `N` is the number of features (e.g. stock tickers).
    
    Returns:
        A tuple containing:
            - List of lists representing the average linkage clustering dendrogram, where each list has the form [left_cluster, right_cluster].
            - Matrix of shape (N, N) containing the distances between clusters in the dendrogram.

    Reference:
        Implementation based on the pyRTM package (https://github.com/GGiecold/pyRMT).
    """
    N = R.shape[0]
    idx = np.triu_indices(N, 1)
    d = R[idx]
    out = fastcluster.average(d)
    rho = 1 - out[:, 2]
    #
    #Genealogy Set
    dend = [([i], []) for i in range(N)] + [[] for _ in range(out.shape[0])]
    
    for i,(a,b) in enumerate(out[:,:2].astype(int)):
        dend[i + N] = dend[a][0] + dend[a][1], dend[b][0] + dend[b][1]
    
    return dend[N:], rho


def average_linkage_clustering(R: pd.DataFrame) -> np.ndarray:
    """
    Performs average linkage clustering on the given correlation matrix `R`.
    
    Args:
        R (pd.DataFrame): Correlation matrix of shape (T, N), where `N` is the number of features (e.g. stock tickers).
    
    Returns:
        np.ndarray: Matrix of shape (N, N) containing the distances between clusters in the clustering dendrogram.
    """
    R = R.corr().fillna(0).values
    N = R.shape[0]
    Rs = np.zeros((N, N))
    Dend, rho = dist(1-R)
    
    for (a,b), r in zip(Dend, rho):
        z = np.array(a).reshape(-1, 1), np.array(b).reshape(1, -1)
        Rs[z] = r
    
    Rs = Rs + Rs.T
    np.fill_diagonal(Rs,1)
    return Rs


########## RANDOM MATRIX THEORY ##########


def marcenkoPastur(X):
    """
       Parameter
       ---------
       X: random matrix of shape (T, N), with T denoting the number
           of samples, whereas N refers to the number of features.
           It is assumed that the variance of the elements of X
           has been normalized to unity.           
       Returns
       -------
       (lambda_min, lambda_max): type tuple
           Bounds to the support of the Marcenko-Pastur distribution
           associated to random matrix X.
       rho: type function
           The Marcenko-Pastur density.
       Reference
       ---------
       "DISTRIBUTION OF EIGENVALUES FOR SOME SETS OF RANDOM MATRICES",
       V. A. Marcenko and L. A. Pastur
       Mathematics of the USSR-Sbornik, Vol. 1 (4), pp 457-483
       Implementation based on the pyRTM package (https://github.com/GGiecold/pyRMT).
    """

    T, N, = X.shape
    q = N / float(T)

    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2

    def rho(x):
        ret = np.sqrt((lambda_max - x) * (x - lambda_min))
        ret /= 2 * np.pi * q * x
        return ret if lambda_min < x < lambda_max else 0.0

    return (lambda_min, lambda_max), rho


def rtm_clipped(X: pd.DataFrame, alpha: Union[List[float], float, Real] = []) -> np.ndarray:
    """
    Clips the eigenvalues of an empirical correlation matrix `E` in order to provide a cleaned estimator `E_clipped` 
    of the underlying correlation matrix.
    Proceeds by keeping the `[N * alpha]` top eigenvalues and shrinking the remaining ones by a trace-preserving constant 


    Args:
        X (pd.DataFrame): Design matrix, of shape (T, N), where `T` denotes the number of samples (think measurements
                          in a time series), while `N` stands for the number of features (think of stock tickers).
        alpha Union[List[float], float, Real], optional): Parameter between 0 and 1, inclusive, determining the fraction to keep of 
                                            the top eigenvalues of an empirical correlation matrix. If left unspecified,
                                            `alpha` is chosen so as to keep all the empirical eigenvalues greater than 
                                            the upper limit of the support to the Marcenko-Pastur spectrum. Defaults to None,
                                            if alpha is a list of floats, the function returns the estimator for all alphas

    Returns:
        np.ndarray: Cleaned estimator of the true correlation matrix 
    
    References:
       "Financial Applications of Random Matrix Theory: a short review",
       J.-P. Bouchaud and M. Potters
       arXiv: 0910.1205 [q-fin.ST]
       Implementation based on the pyRTM package (https://github.com/GGiecold/pyRMT).
    """

    try:
        if alpha is not None:
            if isinstance(alpha, Real):
                alpha = [alpha]
            for a in alpha:
                assert isinstance(a, Real) and 0 <= a <= 1
            
    except AssertionError:
        raise
        sys.exit(1)
    
    T, N = X.shape
        
    E = X.corr().fillna(0).values
    
    eigvals, eigvecs = np.linalg.eigh(E)
    eigvecs = eigvecs.T

    E_clipped = []
    alpha = [None] + alpha
    for a in alpha:
        if a is None:
            (lambda_min, lambda_max), _ = marcenkoPastur(X)
            xi_clipped = np.where(eigvals >= lambda_max, eigvals, np.nan)
        else:
            xi_clipped = np.full(N, np.nan)
            threshold = int(ceil(a * N))
            if threshold > 0:
                xi_clipped[-threshold:] = eigvals[-threshold:]

        gamma = float(E.trace() - np.nansum(xi_clipped))
        gamma /= np.isnan(xi_clipped).sum()
        xi_clipped = np.where(np.isnan(xi_clipped), gamma, xi_clipped)

        E_clipped_i = np.zeros((N, N), dtype=float)
        for xi, eigvec in zip(xi_clipped, eigvecs):
            eigvec = eigvec.reshape(-1, 1)
            E_clipped_i += xi * eigvec.dot(eigvec.T)
        
        tmp = 1./np.sqrt(np.diag(E_clipped_i))
        E_clipped_i *= tmp
        E_clipped_i *= tmp.reshape(-1, 1)

        E_clipped.append(E_clipped_i)
        
    if len(E_clipped) == 1:
        return E_clipped[0]
    else:
        return E_clipped

########## BOOTSTRAPPED ##########

from bahc import filterCovariance

def bootsrap_halc(R):
   """
   Wrapper function for bootsrapped hierarhical clustering. Filters a covariance matrix using the BAHC algorithm.
   
   Args:
       R (pd.DataFrame): Data matrix of shape (T, N).
   
   Returns:
       np.ndarray: Filtered covariance matrix of shape (N, N).
   """
   F = filterCovariance(R.T.fillna(0).values, K=1, Nboot=10, is_correlation=True, method='near')
   return F

########## RIE ##########

def rie_wrapper(R):
    T, N = R.shape
    R_clean = R.dropna(axis=1)
    col_names = {col: i for i, col in enumerate(R.columns)}
    notnans = np.array([col_names[col] for col in set(R_clean.columns)])
    F = np.zeros((N, N))
    C = rie_estimator.get_rie(R_clean)
    F[np.ix_(notnans,notnans)] = C
    np.fill_diagonal(F, 1)
    return F

if __name__ == '__main__':
    pass