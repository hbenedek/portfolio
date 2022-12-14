# Cluster and RTM analysis for portfolio optimization

## Abstract

Portfolio optimization is a technique used by investors to maximize the return on their investment portfolio while
minimizing the risk. It involves selecting a mix of investments that are expected to provide the best combination of
risk and return. This is typically achieved by diversifying the portfolio across a range of assets, such as stocks, bonds,
and commodities, and by carefully considering the trade-off between risk and return for each individual investment.
By using portfolio optimization, investors can create a balanced portfolio that is tailored to their specific investment
goals and risk tolerance. In this report we investigate different techniques on how to solve the optimization problem,
putting emphasis on the statistical uncertainty of the correlation matrix and its cleaning procedures. We try to answer the following questions:

- How does the out-of-sample risk and N_eff evolve for different covariance estimates?
- How does the reliability change in RTM for different cut-off values?
- How does the reliability of the models changes in the function of T and N ?

## Dependencies
You can create a conda environment and use the “requirements.txt” file to install necessary dependencies

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name projectenv
conda activate projectenv
python3 -m pip install -r requirements.txt
```


## Running the code

First step is to download and save the NASDAQ stock prices. This takes a couple of minutes and can be done by running

```bash
python3 run.py --question 0 --start "2010-01-01" --end "2010-12-31"
```

In order to run the model pipeline, execute file `run.py`, to run the computations for Question 1

```bash
python3 run.py --question 1 --T 252 --method average
```

to run the computations for Question 2

```bash
python3 run.py --question 2 --nb_cells 10 --T 300 --N 200
```

to run the computations for Question 3

```bash
python3 run.py --question 3 --max_T 1000 --max_N 500 --Nboot 50 --nb_cells 11
```

the repo contains the saved csv and pickle files for the three questions, the plots can be reproduce by running, where the question flag is 1, 2 or 3

```bash
python3 plot.py --question 1
```

## Repo Structure

<pre>  
├─── run.py : final wrapper script to reproduce results
├─── model.py : RTM and HALC algorithm
├─── preprocess.py : downloading and data preprocessing functions
├─── utils.py : rolling and bootstrap calculations, along with summary statistic calculations
├─── plot.py : auxiliary functions for plotting the results
├─── data
├─── results 
├─── plots 
├─── .gitignore
├─── report.pdf : report containing results, discussions and methodologies
├─── requirements.txt : contains all the dependencies
└─── README.md : README
</pre>
