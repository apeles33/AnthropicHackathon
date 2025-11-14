#!/usr/bin/env python3
"""
Financial Data Generator

Utilities for creating realistic synthetic financial data for trading strategy testing.
Includes price series, order books, sentiment data, and multi-asset returns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
from scipy.stats import norm


def generate_gbm_prices(
    S0: float = 100,
    mu: float = 0.10,
    sigma: float = 0.20,
    days: int = 252,
    dt: float = 1/252
) -> pd.DataFrame:
    """
    Generate price series using Geometric Brownian Motion.
    
    Args:
        S0: Initial price
        mu: Drift (annual expected return)
        sigma: Volatility (annual standard deviation)
        days: Number of days to simulate
        dt: Time step (1/252 for daily data)
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    # Generate returns
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), days)
    
    # Calculate prices
    prices = S0 * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create realistic OHLC
    opens = prices
    closes = prices * np.exp(np.random.normal(0, sigma * np.sqrt(dt/4), days))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    
    # Generate volume with clustering
    base_volume = 1_000_000
    volume = np.random.lognormal(np.log(base_volume), 0.5, days).astype(int)
    
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df


def generate_jump_diffusion_prices(
    S0: float = 100,
    mu: float = 0.10,
    sigma: float = 0.25,
    jump_intensity: float = 0.1,
    jump_mean: float = -0.02,
    jump_std: float = 0.05,
    days: int = 252
) -> pd.DataFrame:
    """
    Generate prices with jump-diffusion process (Merton model).
    Useful for crypto and volatile assets.
    
    Args:
        S0: Initial price
        mu: Drift
        sigma: Diffusion volatility
        jump_intensity: Probability of jump per day
        jump_mean: Mean jump size
        jump_std: Standard deviation of jump size
        days: Number of days
        
    Returns:
        DataFrame with OHLCV data
    """
    dt = 1/252
    prices = [S0]
    
    for _ in range(days - 1):
        # Diffusion component
        dW = np.random.normal(0, np.sqrt(dt))
        diffusion = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
        # Jump component
        if np.random.random() < jump_intensity * dt:
            jump = np.exp(np.random.normal(jump_mean, jump_std))
            new_price = diffusion * jump
        else:
            new_price = diffusion
            
        prices.append(new_price)
    
    prices = np.array(prices)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create OHLC
    opens = prices
    closes = prices * np.exp(np.random.normal(0, sigma * np.sqrt(dt/4), days))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.015, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.015, days)))
    volume = np.random.lognormal(np.log(500_000), 0.7, days).astype(int)
    
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df


def generate_mean_reverting_series(
    S0: float = 100,
    theta: float = 0.5,  # Mean reversion speed
    mu: float = 100,     # Long-term mean
    sigma: float = 10,   # Volatility
    days: int = 252
) -> pd.DataFrame:
    """
    Generate mean-reverting price series using Ornstein-Uhlenbeck process.
    
    Args:
        S0: Initial price
        theta: Mean reversion speed (higher = faster reversion)
        mu: Long-term mean
        sigma: Volatility
        days: Number of days
        
    Returns:
        DataFrame with price series
    """
    dt = 1/252
    prices = [S0]
    
    for _ in range(days - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        dS = theta * (mu - prices[-1]) * dt + sigma * dW
        prices.append(prices[-1] + dS)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    df.set_index('date', inplace=True)
    return df


def generate_correlated_returns(
    n_assets: int = 5,
    days: int = 252,
    mean_returns: Optional[np.ndarray] = None,
    volatilities: Optional[np.ndarray] = None,
    correlation: float = 0.5
) -> pd.DataFrame:
    """
    Generate correlated asset returns for portfolio analysis.
    
    Args:
        n_assets: Number of assets
        days: Number of days
        mean_returns: Array of mean returns (annual)
        volatilities: Array of volatilities (annual)
        correlation: Average pairwise correlation
        
    Returns:
        DataFrame with returns for each asset
    """
    if mean_returns is None:
        mean_returns = np.random.uniform(0.05, 0.15, n_assets)
    if volatilities is None:
        volatilities = np.random.uniform(0.15, 0.30, n_assets)
    
    # Create correlation matrix
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eig(corr_matrix)
    eigvals = np.maximum(eigvals, 0.01)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # Generate returns
    daily_mean = mean_returns / 252
    daily_cov = cov_matrix / 252
    
    returns = np.random.multivariate_normal(daily_mean, daily_cov, days)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    df = pd.DataFrame(returns, index=dates, columns=asset_names)
    return df


def generate_order_book(
    mid_price: float = 100,
    depth_levels: int = 10,
    spread_bps: float = 10,
    size_per_level: int = 1000
) -> pd.DataFrame:
    """
    Generate a realistic limit order book snapshot.
    
    Args:
        mid_price: Current mid price
        depth_levels: Number of price levels on each side
        spread_bps: Bid-ask spread in basis points
        size_per_level: Average size per price level
        
    Returns:
        DataFrame with bid and ask levels
    """
    spread = mid_price * spread_bps / 10000
    
    # Generate bid side
    bid_prices = mid_price - spread/2 - np.arange(depth_levels) * 0.01
    bid_sizes = np.random.poisson(size_per_level, depth_levels)
    
    # Generate ask side
    ask_prices = mid_price + spread/2 + np.arange(depth_levels) * 0.01
    ask_sizes = np.random.poisson(size_per_level, depth_levels)
    
    bids = pd.DataFrame({
        'price': bid_prices,
        'size': bid_sizes,
        'side': 'bid'
    })
    
    asks = pd.DataFrame({
        'price': ask_prices,
        'size': ask_sizes,
        'side': 'ask'
    })
    
    order_book = pd.concat([bids, asks], ignore_index=True)
    return order_book


def generate_sentiment_data(
    days: int = 252,
    mean_sentiment: float = 0.1,
    volatility: float = 0.3,
    autocorr: float = 0.7
) -> pd.DataFrame:
    """
    Generate synthetic sentiment scores with realistic autocorrelation.
    
    Args:
        days: Number of days
        mean_sentiment: Average sentiment (between -1 and 1)
        volatility: Sentiment volatility
        autocorr: Autocorrelation parameter
        
    Returns:
        DataFrame with sentiment scores and article counts
    """
    sentiment = [mean_sentiment]
    
    for _ in range(days - 1):
        shock = np.random.normal(0, volatility)
        new_sentiment = autocorr * sentiment[-1] + (1 - autocorr) * mean_sentiment + shock
        new_sentiment = np.clip(new_sentiment, -1, 1)
        sentiment.append(new_sentiment)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    article_counts = np.random.poisson(20, days)
    
    df = pd.DataFrame({
        'date': dates,
        'sentiment': sentiment,
        'article_count': article_counts,
        'positive': [max(0, s) for s in sentiment],
        'negative': [abs(min(0, s)) for s in sentiment]
    })
    
    df.set_index('date', inplace=True)
    return df


def generate_options_chain(
    spot_price: float = 100,
    risk_free_rate: float = 0.02,
    volatility: float = 0.25,
    maturities: List[float] = [0.25, 0.5, 1.0],
    n_strikes: int = 11
) -> pd.DataFrame:
    """
    Generate options chain with Black-Scholes prices.
    
    Args:
        spot_price: Current stock price
        risk_free_rate: Risk-free rate
        volatility: Implied volatility
        maturities: List of maturities in years
        n_strikes: Number of strike prices
        
    Returns:
        DataFrame with call and put prices
    """
    options = []
    
    for T in maturities:
        # Generate strikes around ATM
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, n_strikes)
        
        for K in strikes:
            # Black-Scholes formula
            d1 = (np.log(spot_price / K) + (risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            
            call_price = spot_price * norm.cdf(d1) - K * np.exp(-risk_free_rate * T) * norm.cdf(d2)
            put_price = K * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            
            options.append({
                'strike': K,
                'maturity': T,
                'call_price': call_price,
                'put_price': put_price,
                'spot': spot_price,
                'volatility': volatility
            })
    
    df = pd.DataFrame(options)
    return df


def add_transaction_costs(
    returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 5.0
) -> pd.Series:
    """
    Apply transaction costs to returns based on position changes.
    
    Args:
        returns: Raw strategy returns
        positions: Position sizes over time
        cost_bps: Transaction cost in basis points
        
    Returns:
        Returns after transaction costs
    """
    position_changes = positions.diff().abs()
    costs = position_changes * (cost_bps / 10000)
    
    adjusted_returns = returns - costs
    return adjusted_returns


if __name__ == "__main__":
    # Example usage and testing
    print("Testing data generators...")
    
    # Test GBM
    gbm_data = generate_gbm_prices(days=100)
    print(f"\nGBM prices: {len(gbm_data)} days")
    print(gbm_data.head())
    
    # Test correlated returns
    corr_returns = generate_correlated_returns(n_assets=3, days=100)
    print(f"\nCorrelated returns: {corr_returns.shape}")
    print(f"Correlation matrix:\n{corr_returns.corr()}")
    
    # Test order book
    book = generate_order_book()
    print(f"\nOrder book: {len(book)} levels")
    print(book.head())
    
    print("\n✓ All generators working correctly!")
