import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf


def get_yf_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData["Close"]
    returns = stockData.pct_change(fill_method=None)
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

historical_data = 50

path = "/Users/leofeingold/Desktop/Financial_Monte_Carlo/sp500holdings.csv"
sp500holdings = pd.read_csv(path)
stockList = sp500holdings["Symbol"].tolist()
stockList = [stock.replace('.B', '-B') for stock in stockList]

sp500holdings["Portfolio%"] = sp500holdings["Portfolio%"].str.replace('%', '').astype(float) / 100
weights = sp500holdings["Portfolio%"].tolist()
weights /= np.sum(weights)
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=historical_data)

meanReturns, covMatrix = get_yf_data(stockList, startDate, endDate)

mc_sims = 1000
T = 100

epsilon = 1e-10
while True:
    try:
        L = np.linalg.cholesky(covMatrix + epsilon * np.eye(len(covMatrix)))
        break
    except np.linalg.LinAlgError:
        epsilon *= 10


meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1) * initialPortfolio

mean_return_after_100_days = portfolio_sims[-1].mean()
median_return = np.median(portfolio_sims[-1])
print(f"Mean return after 100 days: ${mean_return_after_100_days:,.2f}")
print(f"50th Percentile return after 100 days: ${median_return:,.2f}")


def mcVaR(returns, alpha=5):
    # essentially, if 5th percentile performance, how much would you lose... this is value at risk
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a Pandas Series.")

def mcCVaR(returns, alpha=5):
    # what would be the mean loss below the 5th percentile performance
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a Pandas Series.")

portResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

print(f"VaR: ${VaR:,.2f}")
print(f"CVaR: ${CVaR:,.2f}")

plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Days")
plt.suptitle(f"Monte Carlo Simulation of ${initialPortfolio:,} Portfolio Tracking S&P500 Over 100 Days ({mc_sims} Simulations, Trained On Previous {historical_data} Days Of Market Data)")
plt.title(f"50th Percentile: ${median_return:,.2f}, Value At Risk: {VaR:,.2f}, Conditional Value At Risk: {CVaR:,.2f}")
plt.show()


    
