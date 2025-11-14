# Quantitative Finance Code Patterns

## Standard Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Financial libraries
import yfinance as yf  # For real data (optional)
from scipy import stats
from scipy.optimize import minimize

# Set random seed
np.random.seed(42)
```

## Backtest Framework

```python
class Backtest:
    def __init__(self, data, initial_capital=100000):
        self.data = data
        self.initial_capital = initial_capital
        self.positions = []
        self.portfolio_value = []
        self.trades = []
        
    def run(self, strategy_func):
        """Run backtest with given strategy function"""
        capital = self.initial_capital
        position = 0  # Current position size
        
        for i in range(len(self.data)):
            # Get signal from strategy
            signal = strategy_func(self.data.iloc[:i+1])
            
            # Execute trade if signal changed
            if signal != position:
                trade_price = self.data.iloc[i]['close']
                trade_size = signal - position
                
                self.trades.append({
                    'date': self.data.index[i],
                    'price': trade_price,
                    'size': trade_size,
                    'type': 'BUY' if trade_size > 0 else 'SELL'
                })
                
                capital -= trade_size * trade_price
                position = signal
            
            # Calculate portfolio value
            portfolio_val = capital + position * self.data.iloc[i]['close']
            self.portfolio_value.append(portfolio_val)
            self.positions.append(position)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        total_return = (self.portfolio_value[-1] / self.initial_capital - 1)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        max_dd = self.max_drawdown()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades)
        }
    
    def max_drawdown(self):
        """Calculate maximum drawdown"""
        values = pd.Series(self.portfolio_value)
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        return drawdown.min()
```

## Common Strategies

### Moving Average Crossover
```python
def ma_crossover_strategy(data, short_window=20, long_window=50):
    """Simple moving average crossover"""
    data = data.copy()
    data['short_ma'] = data['close'].rolling(short_window).mean()
    data['long_ma'] = data['close'].rolling(long_window).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    
    return data['signal'].iloc[-1]
```

### Mean Reversion (Bollinger Bands)
```python
def bollinger_band_strategy(data, window=20, num_std=2):
    """Mean reversion using Bollinger Bands"""
    data = data.copy()
    data['sma'] = data['close'].rolling(window).mean()
    data['std'] = data['close'].rolling(window).std()
    data['upper'] = data['sma'] + num_std * data['std']
    data['lower'] = data['sma'] - num_std * data['std']
    
    # Generate signals
    current_price = data['close'].iloc[-1]
    if current_price < data['lower'].iloc[-1]:
        return 1  # Buy
    elif current_price > data['upper'].iloc[-1]:
        return -1  # Sell
    else:
        return 0  # Hold
```

### Pairs Trading
```python
def find_cointegrated_pairs(prices_df):
    """Find cointegrated pairs using Engle-Granger test"""
    from statsmodels.tsa.stattools import coint
    
    n = prices_df.shape[1]
    pvalue_matrix = np.ones((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            _, pvalue, _ = coint(prices_df.iloc[:, i], prices_df.iloc[:, j])
            pvalue_matrix[i, j] = pvalue
            pvalue_matrix[j, i] = pvalue
    
    # Find pairs with p-value < 0.05
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if pvalue_matrix[i, j] < 0.05:
                pairs.append((i, j, pvalue_matrix[i, j]))
    
    return pairs
```

## Portfolio Optimization

### Mean-Variance (Markowitz)
```python
def optimize_portfolio(returns, risk_aversion=1.0):
    """
    Markowitz mean-variance optimization
    
    Args:
        returns: DataFrame of asset returns
        risk_aversion: Risk aversion parameter (higher = more risk-averse)
    
    Returns:
        Optimal weights
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    n_assets = len(mean_returns)
    
    # Objective: Maximize return - risk_aversion * variance
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        return -(portfolio_return - risk_aversion * portfolio_variance)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds: 0 <= weight <= 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    init_guess = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, init_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

### Risk Parity
def risk_parity_portfolio(cov_matrix):
    """
    Risk parity: equal risk contribution from each asset
    """
    n_assets = cov_matrix.shape[0]
    
    def objective(weights):
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        # Minimize variance of risk contributions
        target = np.ones(n_assets) / n_assets
        return np.sum((risk_contrib - target * portfolio_vol)**2)
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, init_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x
```

## Risk Metrics

### Value at Risk (VaR)
```python
def calculate_var(returns, confidence=0.95, method='historical'):
    """
    Calculate Value at Risk
    
    Methods: 'historical', 'parametric', 'monte_carlo'
    """
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
    
    elif method == 'parametric':
        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(1 - confidence, mean, std)
    
    elif method == 'monte_carlo':
        # Simulate returns
        mean = returns.mean()
        std = returns.std()
        simulated = np.random.normal(mean, std, 10000)
        var = np.percentile(simulated, (1 - confidence) * 100)
    
    return var

def calculate_cvar(returns, confidence=0.95):
    """Calculate Conditional VaR (Expected Shortfall)"""
    var = calculate_var(returns, confidence, 'historical')
    cvar = returns[returns <= var].mean()
    return cvar
```

### Beta and Greeks
```python
def calculate_beta(asset_returns, market_returns):
    """Calculate asset beta relative to market"""
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

def calculate_greeks_black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks using Black-Scholes
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) 
                 - r * K * np.exp(-r*T) * norm.cdf(d2))
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2))
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }
```

## Time Series Analysis

### ARIMA Forecasting
```python
from statsmodels.tsa.arima.model import ARIMA

def fit_arima(data, order=(1,1,1)):
    """Fit ARIMA model and generate forecast"""
    model = ARIMA(data, order=order)
    fitted = model.fit()
    
    # Forecast
    forecast = fitted.forecast(steps=30)
    
    return fitted, forecast
```

### GARCH Volatility
```python
from arch import arch_model

def fit_garch(returns, p=1, q=1):
    """Fit GARCH model for volatility forecasting"""
    model = arch_model(returns, vol='Garch', p=p, q=q)
    fitted = model.fit(disp='off')
    
    # Forecast volatility
    forecast = fitted.forecast(horizon=30)
    
    return fitted, forecast
```

## Performance Metrics

```python
def calculate_performance_metrics(returns, benchmark_returns=None):
    """Calculate comprehensive performance metrics"""
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns.mean())**252 - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted
    sharpe = (annualized_return - 0.02) / volatility  # Assume 2% risk-free rate
    sortino = annualized_return / (returns[returns < 0].std() * np.sqrt(252))
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    }
    
    # Benchmark comparison
    if benchmark_returns is not None:
        beta = calculate_beta(returns, benchmark_returns)
        excess_returns = returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        metrics['beta'] = beta
        metrics['information_ratio'] = information_ratio
    
    return metrics
```

## Visualization Standards

```python
def plot_backtest_results(returns, benchmark_returns=None):
    """Standard backtest visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    axes[0].plot(cum_returns.index, cum_returns.values, label='Strategy')
    
    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        axes[0].plot(cum_bench.index, cum_bench.values, label='Benchmark', alpha=0.7)
    
    axes[0].set_ylabel('Cumulative Returns')
    axes[0].set_title('Strategy Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    axes[1].set_ylabel('Drawdown')
    axes[1].set_title('Drawdown Analysis')
    axes[1].grid(True, alpha=0.3)
    
    # Rolling Sharpe
    rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values)
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Rolling Sharpe (60-day)')
    axes[2].set_title('Rolling Sharpe Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/backtest_results.png', dpi=300)
    plt.close()
```

## Common Pitfalls

1. **Look-ahead bias**: Don't use future information
2. **Survivorship bias**: Include delisted stocks
3. **Transaction costs**: Always include slippage and commissions
4. **Data snooping**: Be careful of overfitting
5. **Regime changes**: Test across different market conditions
6. **Risk management**: Always use stop-losses
