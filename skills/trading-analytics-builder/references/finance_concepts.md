# Finance Concepts Reference

This document provides quick reference for key financial concepts, formulas, and metrics used in trading analytics projects.

## Performance Metrics

### Return Metrics

**Total Return**
```
R_total = (V_final - V_initial) / V_initial
```

**Annualized Return**
```
R_annual = (1 + R_total)^(365/days) - 1
```

**CAGR (Compound Annual Growth Rate)**
```
CAGR = (V_final / V_initial)^(1/years) - 1
```

**Log Returns**
```
r_t = ln(P_t / P_{t-1})
```
Advantage: Time-additive and normally distributed under GBM

### Risk Metrics

**Volatility (Standard Deviation)**
```
σ = sqrt(Σ(r_i - μ)² / (n-1)) * sqrt(252)
```
Annualization factor: sqrt(252) for daily, sqrt(12) for monthly

**Maximum Drawdown**
```
DD_t = (V_t - max(V_0...V_t)) / max(V_0...V_t)
MDD = min(DD_t)
```

**Value at Risk (VaR)**
Historical: α-percentile of return distribution
Parametric: μ - z_α * σ

**Conditional Value at Risk (CVaR / Expected Shortfall)**
```
CVaR_α = E[R | R < VaR_α]
```
Average of losses exceeding VaR threshold

**Beta**
```
β = Cov(R_asset, R_market) / Var(R_market)
```

### Risk-Adjusted Return Metrics

**Sharpe Ratio**
```
Sharpe = (R_p - R_f) / σ_p
```
R_p: Portfolio return, R_f: Risk-free rate (typically 2%), σ_p: Portfolio volatility

**Sortino Ratio**
```
Sortino = (R_p - R_f) / σ_downside
```
Only penalizes downside volatility

**Calmar Ratio**
```
Calmar = R_annual / |MDD|
```
Return per unit of maximum drawdown risk

**Information Ratio**
```
IR = (R_p - R_b) / TE
```
TE: Tracking error (volatility of excess returns)

**Omega Ratio**
```
Ω(L) = ∫[L,∞] (1 - F(r))dr / ∫[-∞,L] F(r)dr
```
Ratio of probability-weighted gains to losses

### Trade Analysis Metrics

**Win Rate**
```
Win Rate = N_winning_trades / N_total_trades
```

**Profit Factor**
```
PF = Σ(winning_trades) / |Σ(losing_trades)|
```

**Average Win/Loss**
```
Avg Win = Σ(winning_trades) / N_winning_trades
Avg Loss = Σ(losing_trades) / N_losing_trades
```

**Expectancy**
```
E = (Win Rate * Avg Win) - ((1 - Win Rate) * |Avg Loss|)
```

**Kelly Criterion**
```
f* = (p * b - q) / b
```
f*: Optimal fraction to bet, p: win probability, q: loss probability, b: win/loss ratio

## Option Pricing

### Black-Scholes Formula

**Call Price**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Price**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

S₀: Spot price, K: Strike, r: Risk-free rate, T: Time to maturity, σ: Volatility

### Greeks

**Delta (Δ)**: Sensitivity to underlying price
```
Δ_call = N(d₁)
Δ_put = N(d₁) - 1
```

**Gamma (Γ)**: Sensitivity of delta to underlying price
```
Γ = φ(d₁) / (S₀σ√T)
```
φ: Standard normal PDF

**Vega (ν)**: Sensitivity to volatility
```
ν = S₀φ(d₁)√T
```

**Theta (Θ)**: Time decay
```
Θ_call = -S₀φ(d₁)σ/(2√T) - rKe^(-rT)N(d₂)
```

**Rho (ρ)**: Sensitivity to interest rates
```
ρ_call = KTe^(-rT)N(d₂)
```

## Portfolio Theory

### Markowitz Mean-Variance Optimization

**Portfolio Return**
```
R_p = w'μ
```

**Portfolio Variance**
```
σ²_p = w'Σw
```

**Optimization Problem**
```
min w'Σw
subject to: w'μ = R_target, w'1 = 1
```

### Risk Parity

Equal risk contribution from each asset:
```
w_i * (Σw)_i / (√(w'Σw)) = constant for all i
```

### Black-Litterman Model

Combines market equilibrium with investor views:
```
E[R] = [(τΣ)^(-1) + P'ΩP]^(-1) [(τΣ)^(-1)Π + P'Ω Q]
```

## Time Series Models

### ARMA(p,q)

```
y_t = c + Σφ_i y_{t-i} + ε_t + Σθ_j ε_{t-j}
```

### GARCH(1,1)

```
r_t = μ + ε_t
ε_t = σ_t z_t
σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
```

### Ornstein-Uhlenbeck (Mean Reversion)

```
dX_t = θ(μ - X_t)dt + σ dW_t
```
θ: Mean reversion speed, μ: Long-term mean

Half-life of mean reversion:
```
t_{1/2} = ln(2) / θ
```

## Trading Strategy Concepts

### Momentum

Price continuation: assets with high past returns continue to outperform
- Lookback period: 1-12 months
- Skip recent month to avoid reversal
- Equal-weighted or volatility-weighted positions

### Mean Reversion

Price reversal: assets deviate from fundamental value
- Z-score triggers: enter at ±2σ, exit at 0
- Cointegration-based for pairs
- Works best in range-bound markets

### Technical Indicators

**Moving Average**
```
MA_t = (1/n) Σ P_{t-i}
```

**RSI (Relative Strength Index)**
```
RSI = 100 - [100 / (1 + RS)]
RS = Avg_Gain / Avg_Loss
```

**Bollinger Bands**
```
Upper = MA + k*σ
Lower = MA - k*σ
```
Typical k = 2

**MACD**
```
MACD = EMA_12 - EMA_26
Signal = EMA_9(MACD)
```

## Risk Management

### Position Sizing

**Fixed Fractional**
```
Position Size = (Account * Risk%) / Stop Loss Distance
```

**Volatility-Based**
```
Position Size = Target_Vol / Asset_Vol
```

**Kelly Criterion** (see above)

### Stop Loss Methods

1. Fixed percentage (e.g., 2% of capital)
2. ATR-based (e.g., 2x ATR)
3. Support/resistance levels
4. Volatility-adjusted

## Common Pitfalls in Backtesting

### Look-Ahead Bias
Using information not available at the time of decision
- Solution: Point-in-time data, proper splitting

### Survivorship Bias
Only testing on assets that survived to present
- Solution: Include delisted/bankrupt assets

### Overfitting
Strategy works on historical data but not forward
- Solution: Out-of-sample testing, walk-forward analysis

### Data Snooping
Testing many strategies and reporting best result
- Solution: Bonferroni correction, hypothesis testing

### Transaction Costs
Ignoring commissions, slippage, market impact
- Solution: Model realistic costs (5-10 bps for stocks)

### Market Regime Changes
Strategy optimized for one regime fails in another
- Solution: Regime detection, adaptive parameters

## Statistical Tests

### Cointegration (Engle-Granger)

```
1. Estimate: y_t = α + β x_t + ε_t
2. Test: ε_t ~ ADF test
3. If stationary → cointegrated
```

### Stationarity (ADF Test)

```
H0: Unit root (non-stationary)
H1: Stationary
```
Reject H0 if test statistic < critical value

### Normality Tests

- Jarque-Bera test
- Shapiro-Wilk test
- Kolmogorov-Smirnov test

### Autocorrelation

**Ljung-Box Test**
```
Q = n(n+2) Σ [ρ²_k / (n-k)]
```
Tests for serial correlation in residuals

## Market Microstructure

### Bid-Ask Spread Components

1. **Order processing costs**: Dealer compensation
2. **Inventory costs**: Holding risk
3. **Adverse selection**: Informed traders

### Price Impact Models

**Square-root law**
```
Impact ∝ √(Q / V)
```
Q: Order size, V: Daily volume

### Liquidity Measures

- **Bid-ask spread**: Basic liquidity metric
- **Depth**: Size available at best prices  
- **Amihud illiquidity**: |r_t| / Volume_t
- **Roll measure**: -Cov(Δp_t, Δp_{t-1})

## Useful Approximations

**Annualization Factors**
- Daily → Annual: multiply by 252
- Weekly → Annual: multiply by 52
- Monthly → Annual: multiply by 12

**Quick Volatility Scaling**
```
σ(T days) ≈ σ(1 day) * √T
```

**Rule of 72** (doubling time)
```
Years to double ≈ 72 / annual_return_percent
```

**Volatility → Daily Move**
```
20% annual vol ≈ 1.25% daily move (20/√252)
```

## References

Key textbooks and papers:
- Quantitative Trading (Ernest Chan)
- Active Portfolio Management (Grinold & Kahn)
- Market Microstructure Theory (O'Hara)
- Advances in Financial Machine Learning (López de Prado)
