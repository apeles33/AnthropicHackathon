"""
Data Loading Utilities for Research Projects
Provides easy access to common ML datasets and financial data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ===== ML Dataset Loaders =====

def load_mnist(flatten=False):
    """Load MNIST dataset"""
    try:
        from torchvision import datasets, transforms
        import torch
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        if flatten:
            # Flatten for non-CNN models
            train_loader = torch.utils.data.DataLoader(train, batch_size=60000)
            test_loader = torch.utils.data.DataLoader(test, batch_size=10000)
            X_train = next(iter(train_loader))[0].view(60000, -1).numpy()
            y_train = next(iter(train_loader))[1].numpy()
            X_test = next(iter(test_loader))[0].view(10000, -1).numpy()
            y_test = next(iter(test_loader))[1].numpy()
            return (X_train, y_train), (X_test, y_test)
        
        return train, test
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return None, None

def load_cifar10():
    """Load CIFAR-10 dataset"""
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        
        return train, test
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return None, None

def load_fashion_mnist():
    """Load Fashion-MNIST dataset"""
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        
        return train, test
    except Exception as e:
        print(f"Error loading Fashion-MNIST: {e}")
        return None, None

def load_text_corpus(name='imdb', max_samples=None):
    """Load text dataset"""
    try:
        if name == 'imdb':
            from torchtext.datasets import IMDB
            train, test = IMDB(split=('train', 'test'))
            return train, test
        elif name == 'ag_news':
            from torchtext.datasets import AG_NEWS
            train, test = AG_NEWS(split=('train', 'test'))
            return train, test
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None

# ===== Financial Data Functions =====

def simulate_market_data(days=252, frequency='1D', n_assets=1, volatility=0.2, drift=0.1):
    """
    Simulate market data using geometric Brownian motion
    
    Args:
        days: Number of trading days
        frequency: '1D' for daily, '1H' for hourly, '1min' for minute
        n_assets: Number of assets to simulate
        volatility: Annual volatility
        drift: Annual drift (expected return)
    
    Returns:
        DataFrame with OHLCV data
    """
    freq_map = {'1D': 252, '1H': 252*6.5, '1min': 252*6.5*60}
    periods_per_year = freq_map.get(frequency, 252)
    
    if frequency == '1D':
        periods = days
    elif frequency == '1H':
        periods = days * 6.5  # 6.5 trading hours per day
    else:
        periods = days * 6.5 * 60
    
    dt = 1 / periods_per_year
    
    # Generate returns
    returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), (int(periods), n_assets))
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Create OHLCV data
    data = []
    start_date = datetime(2020, 1, 1)
    
    for i in range(int(periods)):
        if frequency == '1D':
            date = start_date + timedelta(days=i)
        elif frequency == '1H':
            date = start_date + timedelta(hours=i)
        else:
            date = start_date + timedelta(minutes=i)
        
        # Simulate intraday variation
        price = prices[i, 0]
        daily_vol = price * 0.01  # 1% intraday variation
        
        high = price + abs(np.random.normal(0, daily_vol))
        low = price - abs(np.random.normal(0, daily_vol))
        open_price = price + np.random.normal(0, daily_vol/2)
        close_price = price
        
        volume = np.random.lognormal(15, 1)  # Realistic volume distribution
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def generate_factor_data(n_assets=100, n_periods=252):
    """
    Generate synthetic factor data for factor modeling
    
    Returns:
        returns: Asset returns (n_periods x n_assets)
        factors: Factor returns (n_periods x n_factors)
        betas: Factor loadings (n_assets x n_factors)
    """
    n_factors = 5
    
    # Generate factor returns
    factor_returns = np.random.normal(0.001, 0.02, (n_periods, n_factors))
    
    # Generate factor loadings (betas)
    betas = np.random.uniform(-1, 1, (n_assets, n_factors))
    
    # Generate asset returns from factors + idiosyncratic risk
    systematic_returns = factor_returns @ betas.T
    idiosyncratic = np.random.normal(0, 0.01, (n_periods, n_assets))
    asset_returns = systematic_returns.T + idiosyncratic
    
    # Create DataFrames
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    returns_df = pd.DataFrame(
        asset_returns.T,
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    factors_df = pd.DataFrame(
        factor_returns,
        index=dates,
        columns=[f'Factor_{i}' for i in range(n_factors)]
    )
    
    betas_df = pd.DataFrame(
        betas,
        index=[f'Asset_{i}' for i in range(n_assets)],
        columns=[f'Factor_{i}' for i in range(n_factors)]
    )
    
    return returns_df, factors_df, betas_df

def simulate_order_book(depth=10, duration_seconds=60):
    """
    Simulate order book data for market microstructure analysis
    
    Args:
        depth: Number of price levels on each side
        duration_seconds: How many seconds of data to generate
    
    Returns:
        List of order book snapshots
    """
    snapshots = []
    mid_price = 100.0
    tick_size = 0.01
    
    for t in range(duration_seconds * 10):  # 10 updates per second
        # Random walk mid price
        mid_price += np.random.choice([-tick_size, 0, tick_size], p=[0.25, 0.5, 0.25])
        
        # Generate bid side
        bids = []
        for i in range(depth):
            price = mid_price - (i + 1) * tick_size
            size = np.random.exponential(100)
            bids.append({'price': price, 'size': size})
        
        # Generate ask side
        asks = []
        for i in range(depth):
            price = mid_price + (i + 1) * tick_size
            size = np.random.exponential(100)
            asks.append({'price': price, 'size': size})
        
        snapshots.append({
            'timestamp': t / 10.0,
            'mid_price': mid_price,
            'bids': bids,
            'asks': asks,
            'spread': asks[0]['price'] - bids[0]['price']
        })
    
    return snapshots

def generate_options_data(spot_price=100, strike_range=(80, 120), n_strikes=20, 
                         days_to_expiry=30, volatility=0.3):
    """
    Generate synthetic options data with Greeks
    
    Returns:
        DataFrame with options chain data
    """
    from scipy.stats import norm
    
    strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)
    T = days_to_expiry / 365.0
    r = 0.05  # Risk-free rate
    
    data = []
    for K in strikes:
        # Black-Scholes calculations
        d1 = (np.log(spot_price / K) + (r + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        # Call option
        call_price = spot_price * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        call_delta = norm.cdf(d1)
        
        # Put option
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        put_delta = -norm.cdf(-d1)
        
        # Greeks (same for both)
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(T))
        vega = spot_price * norm.pdf(d1) * np.sqrt(T)
        
        data.append({
            'strike': K,
            'call_price': call_price,
            'call_delta': call_delta,
            'put_price': put_price,
            'put_delta': put_delta,
            'gamma': gamma,
            'vega': vega,
            'days_to_expiry': days_to_expiry,
            'implied_volatility': volatility
        })
    
    return pd.DataFrame(data)

# ===== Convenience Functions =====

def create_train_val_split(data, val_ratio=0.2, seed=42):
    """Create train/validation split"""
    np.random.seed(seed)
    n = len(data)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - val_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return train_idx, val_idx

def normalize_data(X_train, X_test):
    """Normalize features to zero mean and unit variance"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm

# ===== Example Usage =====

if __name__ == "__main__":
    # ML example
    print("Loading MNIST...")
    train, test = load_mnist()
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Quant example
    print("\nGenerating market data...")
    prices = simulate_market_data(days=252, frequency='1D')
    print(f"Generated {len(prices)} days of price data")
    print(prices.head())
