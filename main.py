import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# tickers
tickers = ["GOOG", "AAPL", "META", "BABA", "AMZN", "GE", "AMD", "WMT",
           "BAC", "GM", "T", "UAA", "XOM", "RRC", "BBY", "MA", "PFE", "JPM", "SBUX"]

# yahoo finance stock price data
def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.dropna()
    return data

def get_current_prices(tickers):
    """Fetch current stock prices for the tickers using yfinance"""
    data = yf.download(tickers, period="1d")['Adj Close']
    return data.iloc[0]  # retrieve most recent data

# read in data
df = download_stock_data(tickers, start_date="2015-01-01", end_date="2023-10-01")
returns = df.pct_change().dropna()

# check for missing or infinite values in returns
if not np.isfinite(returns.values).all():
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

# compute mean historical returns and covariance matrix
mu = returns.mean() * 252  # annualize returns for assumption of 252 trading days
cov_matrix = returns.cov() * 252  # annualize covariance matrix

# efficient frontier using Sharpe Ratio
def portfolio_performance(weights, mu, cov_matrix, risk_free_rate=0.02):
    port_return = np.dot(weights, mu)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def max_sharpe_ratio(mu, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mu)
    
    def neg_sharpe(weights):
        return -portfolio_performance(weights, mu, cov_matrix, risk_free_rate)[2]
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1. / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Black-Litterman Model
def black_litterman_prior(mcaps, delta, cov_matrix):
    market_caps = np.array([mcaps[ticker] for ticker in tickers])
    total_market_cap = np.sum(market_caps)
    weights_market = market_caps / total_market_cap
    pi = delta * np.dot(cov_matrix, weights_market)
    return pi

# Allocate funds based on weights and current prices
def allocate_funds(total_value, weights, prices):
    investments = {}
    total_invested = 0
    
    for i, ticker in enumerate(tickers):
        # calculate investment per stock based on its weight
        invest_amount = weights[i] * total_value
        
        # calculate the number of shares you can buy with the allocated money
        shares = np.floor(invest_amount / prices[ticker])
        
        # calculate how much money will be invested in this stock
        invested = shares * prices[ticker]
        
        investments[ticker] = {"Shares": shares, "Invested": invested}
        total_invested += invested

    leftover = total_value - total_invested
    return investments, total_invested, leftover

# Hierarchical Risk Parity (HRP)
def hrp_portfolio(returns):
    corr_matrix = returns.corr()
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    dist_array = dist_matrix.to_numpy()

    # non-finite values in the distance matrix
    if not np.isfinite(dist_array).all():
        dist_array = np.nan_to_num(dist_array, nan=1, posinf=1, neginf=1)

    # Create clusters
    clusters = linkage(dist_array, method='single')

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(clusters, labels=returns.columns)
    plt.show()

    # Allocate weights (simplified)
    num_assets = len(returns.columns)
    weights = np.ones(num_assets) / num_assets
    return weights

# input total portfolio value
total_portfolio_value = float(input("Enter the total value of your portfolio (in dollars): "))

# Fetch current prices
current_prices = get_current_prices(tickers)

# Run EF
weights_ef = max_sharpe_ratio(mu, cov_matrix)
port_return_ef, port_volatility_ef, sharpe_ef = portfolio_performance(weights_ef, mu, cov_matrix)
investments_ef, total_invested_ef, leftover_ef = allocate_funds(total_portfolio_value, weights_ef, current_prices)

# 2. Black-Litterman Portfolio
mcaps = {
    "GOOG": 927e9, "AAPL": 1.19e12, "META": 574e9, "BABA": 533e9, "AMZN": 867e9,
    "GE": 96e9, "AMD": 43e9, "WMT": 339e9, "BAC": 301e9, "GM": 51e9, "T": 61e9,
    "UAA": 78e9, "XOM": 295e9, "RRC": 1e9, "BBY": 22e9, "MA": 288e9, "PFE": 212e9, 
    "JPM": 422e9, "SBUX": 102e9
}
delta = 2.5  # Market risk aversion (assumed value)
prior = black_litterman_prior(mcaps, delta, cov_matrix)

# Views for
views = np.array([-0.20, 0.10, 0.15]).reshape(-1, 1)
picking_matrix = np.zeros((3, len(tickers)))

picking_matrix[0, tickers.index("SBUX")] = 1
picking_matrix[1, tickers.index("GOOG")] = 1
picking_matrix[1, tickers.index("META")] = -1
picking_matrix[2, tickers.index("BAC")] = 0.5
picking_matrix[2, tickers.index("JPM")] = 0.5
picking_matrix[2, tickers.index("T")] = -0.5
picking_matrix[2, tickers.index("GE")] = -0.5

tau = 0.01  # Uncertainty scale
sub_a = np.linalg.inv(np.dot(np.dot(picking_matrix, tau * cov_matrix), picking_matrix.T) + np.diag(np.diag(np.dot(picking_matrix, np.dot(tau * cov_matrix, picking_matrix.T)))))
adjusted_mean = prior + np.dot(np.dot(tau * cov_matrix, picking_matrix.T), np.dot(sub_a, views - np.dot(picking_matrix, prior).reshape(-1,1))).flatten()

weights_bl = max_sharpe_ratio(adjusted_mean, cov_matrix)
port_return_bl, port_volatility_bl, sharpe_bl = portfolio_performance(weights_bl, adjusted_mean, cov_matrix)
investments_bl, total_invested_bl, leftover_bl = allocate_funds(total_portfolio_value, weights_bl, current_prices)

#run HRP
weights_hrp = hrp_portfolio(returns)
port_return_hrp, port_volatility_hrp, sharpe_hrp = portfolio_performance(weights_hrp, mu, cov_matrix)
investments_hrp, total_invested_hrp, leftover_hrp = allocate_funds(total_portfolio_value, weights_hrp, current_prices)

#analysis function
def analyze_and_recommend():
    strategies = {
        "Efficient Frontier": {"return": port_return_ef, "volatility": port_volatility_ef, "sharpe": sharpe_ef, "leftover": leftover_ef},
        "Black-Litterman": {"return": port_return_bl, "volatility": port_volatility_bl, "sharpe": sharpe_bl, "leftover": leftover_bl},
        "HRP": {"return": port_return_hrp, "volatility": port_volatility_hrp, "sharpe": sharpe_hrp, "leftover": leftover_hrp},
    }

    # output analysis
    print("\nPortfolio Analysis:")
    for strategy, stats in strategies.items():
        print(f"\n{strategy} Strategy:")
        print(f"  Expected Return: {stats['return']:.2%}")
        print(f"  Volatility: {stats['volatility']:.2%}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}")
        print(f"  Leftover Cash: ${stats['leftover']:.2f}")
    
    # reconmend based off sharpe ratio
    best_strategy = max(strategies, key=lambda s: strategies[s]["sharpe"])
    print(f"\nRecommended Strategy (based off Sharpe Ratio): {best_strategy}")

# output results for each stratergy
print(f"\nEfficient Frontier Portfolio:")
print(f"Expected annual return: {port_return_ef:.2%}, Volatility: {port_volatility_ef:.2%}, Sharpe Ratio: {sharpe_ef:.2f}")
print(f"Total Invested: ${total_invested_ef:.2f}, Leftover: ${leftover_ef:.2f}")

print(f"\nBlack-Litterman Portfolio:")
print(f"Expected annual return: {port_return_bl:.2%}, Volatility: {port_volatility_bl:.2%}, Sharpe Ratio: {sharpe_bl:.2f}")
print(f"Total Invested: ${total_invested_bl:.2f}, Leftover: ${leftover_bl:.2f}")

print(f"\nHRP Portfolio:")
print(f"Expected annual return: {port_return_hrp:.2%}, Volatility: {port_volatility_hrp:.2%}, Sharpe Ratio: {sharpe_hrp:.2f}")
print(f"Total Invested: ${total_invested_hrp:.2f}, Leftover: ${leftover_hrp:.2f}")

analyze_and_recommend()
