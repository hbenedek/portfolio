# Cluster and RMT analysis for portfolio optimization

## Abstract

Portfolio optimization is a technique used by investors to maximize the return on their investment portfolio while
minimizing the risk. It involves selecting a mix of investments that are expected to provide the best combination of
risk and return. This is typically achieved by diversifying the portfolio across a range of assets, such as stocks, bonds,
and commodities, and by carefully considering the trade-off between risk and return for each individual investment.
By using portfolio optimization, investors can create a balanced portfolio that is tailored to their specific investment
goals and risk tolerance. In this report we investigate different techniques on how to solve the optimization problem,
putting emphasis on the statistical uncertainty of the correlation matrix and its cleaning procedures. We try to answer the following questions:

- Q1: How does the out-of-sample risk changes for different shrinkage coefficients?
- Q2: Does higher order clustering and bootsrapping improve HALC?
- Q3: How does the reliability change in RMT for different cut-off values?
- Q4: How does the out-of-sample risk and Neff evolve for different covariance estimates?
- Q5: How does the reliability of RMT and HALC changes in the function of T and N ?

## Dependencies
Install the packages: `pip install -r requirements.txt`

- [rie-estimator](https://pypi.org/project/rie-estimator/)
- [fastcluster](https://pypi.org/project/fastcluster/)
- [yfinance](https://pypi.org/project/yfinance/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [seaborn](https://pypi.org/project/seaborn/)
- [tqdm](https://pypi.org/project/tqdm/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [bahc](https://pypi.org/project/bahc/)
- [pyRTM](https://pypi.org/project/pyrtm/)

## Running the code

First step is to download and save the NASDAQ stock prices. This takes a couple of minutes and can be done by running

```bash
python3 run.py --question 0 --start "2010-01-01" --end "2020-12-31"
```

In order to run the model pipeline, execute file `run.py`, different questions can be run with the following commands


```bash
python3 run.py --question 1 --T 300 --N 200 --nb_cells 10 
```

```bash
python3 run.py --question 2 --T 300 --N 200 
```

```bash
python3 run.py --question 3 --T 300 --N 200 --nb_cells 10
```

```bash
python3 run.py --question 4 --T 252 --method clustering
```

```bash
python3 run.py --question 5 --max_T 1000 --max_N 500 --Nboot 50 --nb_cells 11
```

## Repo Structure

<pre>  
├─── run.py               : script to reproduce results, along with rolling window calculations
├─── model.py             : RTM, shrinkage, HALC, BAHC, RIE algorithms
├─── preprocess.py        : downloading and data preprocessing functions
├─── utils.py             : summary statistics calculations
├─── visualizations.ipynb : notebook produces the plots presented in the report
|
├─── data
├─── results 
├─── plots 
|
├─── .gitignore
├─── report.pdf           : report contains results, discussions and methodologies
├─── requirements.txt     : contains all the dependencies
└─── README.md 
</pre>
