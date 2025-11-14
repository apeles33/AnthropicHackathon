---
name: trading-analytics-builder
description: Builds diverse trading and quantitative finance projects across 15 types including momentum trading, mean reversion, sentiment analysis, options pricing, portfolio optimization, market making, arbitrage scanning, risk dashboards, technical indicators, backtesting engines, market microstructure, pairs trading, volatility forecasting, high-frequency simulation, and multi-asset allocation. Each project produces working Python code, backtest results, visualizations, and professional HTML presentations. Use when building trading strategy demonstrations, quantitative finance projects, or educational finance examples.
---

# Trading Analytics Builder

## Overview

Build working trading and quantitative finance projects that implement real algorithms from academic literature and industry practice. Each project follows a research-to-implementation pattern: select concepts from literature, implement core algorithms with simplified but functional code, run experiments with realistic synthetic or historical data, and produce standalone HTML presentations with code, visualizations, and learning outcomes.

## Project Execution Pattern

All projects follow this standardized workflow:

### 1. Concept Selection
- Identify relevant academic papers, textbooks, or industry white papers
- Choose specific algorithms or strategies to implement
- Simplify complex methods while preserving core principles
- Reference seminal works and recent developments

### 2. Environment Setup
```python
# Standard trading analytics stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from datetime import datetime, timedelta
import yfinance as yf  # For historical data when appropriate
```

### 3. Core Algorithm Implementation
- Write clean, well-commented Python code
- Focus on educational clarity over production optimization
- Include parameter configuration and validation
- Implement realistic data generation when using synthetic data

### 4. Experimentation & Backtesting
- Generate or fetch appropriate datasets (synthetic for demos, real for serious analysis)
- Run strategy simulations with proper risk controls
- Calculate standard performance metrics (Sharpe, max drawdown, win rate, etc.)
- Compare against naive baselines

### 5. Visualization & Analysis
- Create clear, publication-quality charts using matplotlib/seaborn
- Include performance curves, distribution plots, and statistical summaries
- Show parameter sensitivity where relevant
- Visualize strategy behavior across market conditions

### 6. HTML Output Generation
Use the provided HTML template (`assets/project_template.html`) to create standalone presentations containing:
- **Title & Description**: Project name and one-paragraph summary
- **Theoretical Foundation**: Brief explanation of underlying concepts and cited papers
- **Code Snippets**: Key implementation details (10-30 lines per section)
- **Results Visualization**: Embedded charts showing strategy performance
- **Performance Metrics**: Table of key statistics (returns, Sharpe, drawdown, etc.)
- **Learning Outcomes**: 3-5 key takeaways about the strategy and its behavior

## Available Project Types

### 1. Momentum Bot
**Concept**: Trend-following strategy based on price momentum and moving average crossovers.

**Implementation**: 
- Calculate exponential moving averages (fast and slow periods)
- Generate signals on crossover events
- Include position sizing based on volatility (ATR)
- Add stop-loss and take-profit levels

**Data**: Use crypto or stock data with sufficient volatility
**Key Papers**: Jegadeesh & Titman (1993), Carhart (1997)

### 2. Mean Reversion Strategy
**Concept**: Buy oversold assets and sell overbought ones based on statistical deviation from mean.

**Implementation**:
- Calculate z-scores or Bollinger Bands
- Detect statistical extremes
- Model mean-reverting behavior with Ornstein-Uhlenbeck process
- Include regime detection to avoid trending markets

**Data**: Pairs of correlated assets or single mean-reverting series
**Key Papers**: Lo & MacKinlay (1988), Gatev et al. (2006)

### 3. Sentiment Analyzer
**Concept**: Transform news sentiment into trading signals using NLP and sentiment scoring.

**Implementation**:
- Use VADER or FinBERT for sentiment scoring (simplified mock in demo)
- Aggregate sentiment scores across multiple sources
- Generate trading signals based on sentiment thresholds
- Include sentiment momentum and changes

**Data**: Simulated news headlines with sentiment scores
**Key Papers**: Tetlock (2007), Loughran & McDonald (2011)

### 4. Options Pricer
**Concept**: Black-Scholes model for European options with Greeks calculation.

**Implementation**:
- Implement closed-form Black-Scholes formula
- Calculate all Greeks (Delta, Gamma, Vega, Theta, Rho)
- Create volatility surface visualization
- Include implied volatility solver (Newton-Raphson)

**Data**: Range of strikes and maturities
**Key Papers**: Black & Scholes (1973), Merton (1973)

### 5. Portfolio Optimizer
**Concept**: Modern portfolio theory with Markowitz mean-variance optimization.

**Implementation**:
- Calculate efficient frontier
- Optimize for maximum Sharpe ratio
- Include constraints (long-only, sector limits, etc.)
- Compare equally-weighted, risk parity, and optimal portfolios

**Data**: Historical returns of 5-15 assets
**Key Papers**: Markowitz (1952), Sharpe (1964)

### 6. Market Maker Simulator
**Concept**: Provide liquidity by quoting bid-ask spreads and managing inventory risk.

**Implementation**:
- Implement Avellaneda-Stoikov model for optimal quotes
- Manage inventory with mean reversion penalties
- Handle adverse selection and order flow toxicity
- Calculate P&L from spread capture vs. inventory risk

**Data**: Simulated order flow with varying arrival rates
**Key Papers**: Avellaneda & Stoikov (2008), Guilbaud & Pham (2013)

### 7. Arbitrage Scanner
**Concept**: Detect and exploit price discrepancies across exchanges or related instruments.

**Implementation**:
- Monitor price differences for same asset across venues
- Calculate arbitrage profit after fees and slippage
- Model triangular arbitrage in currency/crypto markets
- Include execution timing and latency constraints

**Data**: Synthetic multi-exchange price feeds
**Key Papers**: Foucault et al. (2013), Makarov & Schoar (2020)

### 8. Risk Dashboard
**Concept**: Comprehensive portfolio risk metrics including VaR, CVaR, and Greeks exposure.

**Implementation**:
- Calculate Value-at-Risk using historical and parametric methods
- Compute Conditional Value-at-Risk (Expected Shortfall)
- Display Greeks exposure for options portfolios
- Show correlation matrices and factor exposures

**Data**: Portfolio of stocks, options, and bonds
**Key Papers**: Jorion (2006), Rockafellar & Uryasev (2000)

### 9. Technical Indicators Library
**Concept**: Collection of popular technical indicators with visualization and backtesting.

**Implementation**:
- Implement RSI, MACD, Stochastic Oscillator, Bollinger Bands, ADX
- Create combined indicator signals
- Backtest simple strategies based on each indicator
- Visualize indicator behavior and strategy performance

**Data**: Historical OHLCV data
**Key Papers**: Wilder (1978), Murphy (1999)

### 10. Backtest Engine
**Concept**: Event-driven backtesting framework with realistic execution simulation.

**Implementation**:
- Build modular backtester with Strategy/Portfolio/Execution classes
- Include transaction costs, slippage, and market impact
- Calculate standard performance metrics and generate tear sheets
- Support both vectorized and event-driven backtesting

**Data**: Historical price data for multiple assets
**Key Papers**: Pardo (2008), Bailey & López de Prado (2014)

### 11. Market Microstructure Analyzer
**Concept**: Analyze order book dynamics, price impact, and liquidity measures.

**Implementation**:
- Reconstruct limit order book from trades and quotes
- Calculate bid-ask spread components (adverse selection, inventory, processing)
- Measure price impact of large orders
- Visualize order flow and market depth

**Data**: Simulated order book with realistic dynamics
**Key Papers**: Kyle (1985), Glosten & Milgrom (1985), Hasbrouck (2007)

### 12. Pairs Trading Strategy
**Concept**: Statistical arbitrage based on cointegration between related assets.

**Implementation**:
- Test for cointegration using Engle-Granger or Johansen tests
- Calculate hedge ratios and spread z-scores
- Generate trading signals on spread divergence/convergence
- Include half-life calculation for mean reversion speed

**Data**: Pairs of historically correlated stocks or ETFs
**Key Papers**: Vidyamurthy (2004), Pole (2007)

### 13. Volatility Forecaster
**Concept**: GARCH models for volatility prediction with trading applications.

**Implementation**:
- Fit GARCH(1,1) model to returns data
- Forecast future volatility with confidence intervals
- Compare realized vs. implied volatility
- Trade volatility using straddles or variance swaps

**Data**: Historical returns with volatility clustering
**Key Papers**: Engle (1982), Bollerslev (1986)

### 14. High-Frequency Trading Simulator
**Concept**: Latency-aware trading with order placement/cancellation strategies.

**Implementation**:
- Model market at microsecond resolution
- Implement latency arbitrage and queue position strategies
- Include adverse selection from informed traders
- Simulate co-location advantages

**Data**: Synthetic limit order book with nanosecond timestamps
**Key Papers**: Budish et al. (2015), Baron et al. (2019)

### 15. Multi-Asset Allocator
**Concept**: Tactical asset allocation across stocks, bonds, commodities, and crypto.

**Implementation**:
- Implement risk parity allocation
- Add momentum and trend overlays
- Include dynamic rebalancing rules
- Compare constant-mix vs. adaptive strategies

**Data**: Historical returns for 4+ asset classes
**Key Papers**: Bridgewater Associates (2015), Asness et al. (2012)

## Bundled Resources

### scripts/
**`generate_project.py`**: Main script that orchestrates project generation
- Takes project type as input (1-15)
- Generates all code, runs backtests, creates visualizations
- Outputs standalone HTML file

**`data_generator.py`**: Utility for creating realistic synthetic financial data
- Geometric Brownian Motion for price series
- Jump-diffusion processes for crypto/volatile assets
- Correlated multi-asset returns
- Realistic order book dynamics

### references/
**`finance_concepts.md`**: Reference guide with:
- Key financial formulas and equations
- Standard performance metrics definitions
- Risk management principles
- Common pitfalls in backtesting

**`paper_sources.md`**: Curated list of seminal papers for each project type with links and brief summaries

### assets/
**`project_template.html`**: Professional HTML template with:
- Responsive design using Tailwind CSS
- Chart.js integration for interactive plots
- Syntax highlighting for code snippets
- Clean typography and layout

## Usage Examples

**Example 1: Generate Momentum Trading Bot**
```
User: "Build me a momentum trading bot that trades Bitcoin using moving average crossovers"
Claude: [Reads this skill, creates project using type 1 pattern, generates HTML output]
```

**Example 2: Create Options Pricing Dashboard**
```
User: "I want to understand Black-Scholes and see how option prices change with volatility"
Claude: [Uses type 4 pattern, implements B-S with Greeks, creates interactive visualizations]
```

**Example 3: Build Complete Backtesting Framework**
```
User: "Create a professional backtesting engine that I can use for multiple strategies"
Claude: [Implements type 10 pattern with modular design and comprehensive documentation]
```

## Implementation Guidelines

### Code Quality Standards
- Write production-quality Python with type hints where appropriate
- Include docstrings for all functions and classes
- Use meaningful variable names that match financial terminology
- Add inline comments explaining financial logic, not just code mechanics

### Data Handling
- Use pandas for all time series operations
- Ensure proper datetime indexing
- Handle missing data and market holidays appropriately
- Include data validation checks

### Visualization Best Practices
- Use consistent color schemes (green for profits, red for losses)
- Add clear axis labels with units
- Include legends and titles on all charts
- Export charts as base64-encoded images for HTML embedding

### Performance Metrics
Always include these standard metrics:
- **Returns**: Total return, annualized return, CAGR
- **Risk**: Volatility, max drawdown, max drawdown duration
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Statistical**: Win rate, profit factor, average win/loss
- **Trade Analysis**: Number of trades, average holding period

### Educational Value
Each project should teach:
1. **Core Concept**: What is the fundamental idea?
2. **Implementation Details**: How is it coded?
3. **Practical Considerations**: What matters in real trading?
4. **Limitations**: What doesn't work about this approach?
5. **Extensions**: How could this be improved?

## Output Specifications

The final HTML file must include:

1. **Header Section**
   - Project title
   - One-sentence description
   - Date generated
   - Technology stack used

2. **Theory Section**
   - 2-3 paragraph explanation of underlying concepts
   - Links to relevant papers (2-4)
   - Mathematical formulation (if applicable)

3. **Implementation Section**
   - Core algorithm code (20-40 lines)
   - Key function definitions
   - Configuration parameters

4. **Results Section**
   - Performance summary table
   - Equity curve chart
   - Drawdown chart
   - Distribution of returns
   - Additional strategy-specific visualizations

5. **Analysis Section**
   - Interpretation of results
   - Comparison to baseline (buy-and-hold or equally-weighted)
   - Discussion of parameter sensitivity
   - Market regime analysis (if applicable)

6. **Learning Outcomes**
   - 3-5 bullet points of key insights
   - Practical applications
   - Common pitfalls to avoid
   - Suggestions for further exploration

## Quality Checklist

Before delivering any project, verify:
- [ ] Code runs without errors
- [ ] All metrics are calculated correctly
- [ ] Visualizations are clear and informative
- [ ] HTML renders properly with all assets embedded
- [ ] Citations are accurate and accessible
- [ ] Learning outcomes are substantive and educational
- [ ] Results are realistic (no >1000% Sharpe ratios from bugs)
- [ ] Transaction costs and slippage are included where appropriate
