import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_sp500_holdings():
    driver = webdriver.Safari()
    url = "https://www.slickcharts.com/sp500"
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'table-responsive')))

    table = driver.find_element(By.CLASS_NAME, 'table-responsive')
    headers = table.find_elements(By.TAG_NAME, 'th')
    header_names = [header.text for header in headers]

    rows = table.find_elements(By.TAG_NAME, 'tr')
    data = []

    for row in rows[1:]: 
        cells = row.find_elements(By.TAG_NAME, 'td')
        cell_data = [cell.text for cell in cells]
        data.append(cell_data)

    df = pd.DataFrame(data, columns=header_names)
    driver.quit()
    print(df)
    df.to_csv("sp500holdings.csv")

def get_yf_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData["Close"]
    returns = stockData.pct_change(fill_method=None)
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


path = "/Users/leofeingold/Desktop/Financial_Monte_Carlo/sp500holdings.csv"
sp500holdings = pd.read_csv(path)
stockList = sp500holdings["Symbol"].tolist()
stockList = [stock.replace('.B', '-B') for stock in stockList]

sp500holdings["Portfolio%"] = sp500holdings["Portfolio%"].str.replace('%', '').astype(float) / 100
weights = sp500holdings["Portfolio%"].tolist()
weights /= np.sum(weights)
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=150)

meanReturns, covMatrix = get_yf_data(stockList, startDate, endDate)

mc_sims = 100
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

plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value ($")
plt.xlabel("Days")
plt.suptitle(f"Monte Carlo Simulation of ${initialPortfolio:,} Portfolio Tracking S&P500 Over 100 Days ({mc_sims} Simulations)")
plt.title(f"50th Percentile return after 100 days: ${median_return:,.2f}")
plt.show()