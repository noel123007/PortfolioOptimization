import os
import quandl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)

plt.switch_backend("TkAgg")
plt.style.use('fivethirtyeight')
np.random.seed(777)

quandl.ApiConfig.api_key = '84qJQFf5dTyzjvyxAAyM'
stocks = ['AAPL', 'AMZN', 'GOOGL', 'FB']
data = quandl.get_table('WIKI/PRICES', ticker=stocks,
                        qopts={'columns': ['date', 'ticker', 'adj_close']},
                        date={'gte': '2017-1-1', 'lte': '2019-12-31'}, paginate=True)


df = data.set_index('date')
table = df.pivot(columns='ticker')
table.columns = [col[1] for col in table.columns]


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    calc = np.sum(mean_returns * weights)
    returns = calc * 252
    std = np.sqrt(np.dot(weights.T, np.dot(
        cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
            weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        # Sharpe Ratio
    return results, weights_record


returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178

# Allocation weights of based on Minimum Volaitility

alloc_min_A = 0
alloc_min_B = 0
alloc_min_C = 0
alloc_min_D = 0

# Allocation weights of based on Sharpe Ratio

alloc_sr_A = 0
alloc_sr_B = 0
alloc_sr_C = 0
alloc_sr_D = 0


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(
        weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [
        round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(
        weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [
        round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    # Extracting allocation weight for Minimum Volatility
    global alloc_min_A
    global alloc_min_B
    global alloc_min_C
    global alloc_min_D

    alloc_min_A = int(min_vol_allocation.at['allocation', 'AAPL'])
    alloc_min_B = int(min_vol_allocation.at['allocation', 'AMZN'])
    alloc_min_C = int(min_vol_allocation.at['allocation', 'GOOGL'])
    alloc_min_D = int(min_vol_allocation.at['allocation', 'FB'])
    # Extracting allocation weight for Sharpe Ratio
    global alloc_sr_A
    global alloc_sr_B
    global alloc_sr_C
    global alloc_sr_D

    alloc_sr_A = int(max_sharpe_allocation.at['allocation', 'AAPL'])
    alloc_sr_B = int(max_sharpe_allocation.at['allocation', 'AMZN'])
    alloc_sr_C = int(max_sharpe_allocation.at['allocation', 'GOOGL'])
    alloc_sr_D = int(max_sharpe_allocation.at['allocation', 'FB'])


# Taking in principle amount
# print("\n\n\n")
# print("-" * 80)
# print('Enter the principle amount you would like to invest in USD \n')

# print("-" * 80)


# # Allocation of principle amount based on minumum volatility

# princealloc_min_A = int((alloc_min_A/100) * principle)
# princealloc_min_B = int((alloc_min_B/100) * principle)
# princealloc_min_C = int((alloc_min_C/100) * principle)
# princealloc_min_D = int((alloc_min_D/100) * principle)

# Allocation of principle amount based on Sharpe Ratio


# # Displaying the capital allocation for the Portfolio obtained based on minimum volatility
# print("\n\n\n")
# print("-" * 80)
# print("Minimum Volatility Portfolio Capital Allocation \n")
# print("AAPL", end=' : ')
# print(princealloc_min_A, "USD")
# print("AMZN", end=' : ')
# print(princealloc_min_B, "USD")
# print("GOOGL", end=' : ')
# print(princealloc_min_C, "USD")
# print("FB", end=' : ')
# print(princealloc_min_D, "USD")
# print("\n")
# print("-" * 80)


# # Displaying the capital allocation for the Portfolio obtained based on Sharpe Ratio
# print("\n\n\n")
# print("-" * 80)
# print("Sharpe Ratio Portfolio Capital Allocation \n")
# print("AAPL", end=' : ')
# print(princealloc_sr_A, "USD")
# print("AMZN", end=' : ')
# print(princealloc_sr_B, "USD")
# print("GOOGL", end=' : ')
# print(princealloc_sr_C, "USD")
# print("FB", end=' : ')
# print(princealloc_sr_C, "USD")
# print("\n")
# print("-" * 80)


# # Displaying the capital allocation for the Benchmark Portfolio
# print("\n\n\n")
# print("-" * 80)
# print("Benchmark Portfolio Capital Allocation \n")
# print("AAPL", end=' : ')
# print(principle//4, "USD")
# print("AMZN", end=' : ')
# print(principle//4, "USD")
# print("GOOGL", end=' : ')
# print(principle//4, "USD")
# print("FB", end=' : ')
# print(principle//4, "USD")
# print("\n")
# print("-" * 80)


@app.route("/",  methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if not request.form['principleAmount']:
            return render_template("index.html", ptype="empty", message="Cannot be empty!")
        else:
            principleAmount = request.form['principleAmount']
            display_simulated_ef_with_random(
                mean_returns, cov_matrix, num_portfolios, risk_free_rate)
            principleAmount = int(principleAmount)
            princealloc_sr_A = int((alloc_sr_A/100) * principleAmount)
            princealloc_sr_B = int((alloc_sr_B/100) * principleAmount)
            princealloc_sr_C = int((alloc_sr_C/100) * principleAmount)
            princealloc_sr_D = int((alloc_sr_D/100) * principleAmount)
            return render_template("index.html", ptype="submitted", princealloc_sr_A=princealloc_sr_A, princealloc_sr_B=princealloc_sr_B)
    else:
        return render_template('index.html')
