---
name: quant-research-builder
description: Builds diverse quantitative finance research projects across 10 types including factor models (Fama-French, momentum, value), portfolio optimization (Markowitz, Black-Litterman, risk parity), risk analytics (VaR, CVaR, stress testing), market microstructure (order book dynamics, liquidity), options pricing (Black-Scholes, Greeks, volatility surfaces), algorithmic trading strategies (momentum, mean reversion, pairs trading), time series forecasting (ARIMA, GARCH, LSTM), volatility modeling, high-frequency analysis, and quantamental research. Each project produces working Python code, backtest results, visualizations, and professional HTML presentations. Use when building quantitative finance hackathon projects, trading strategy demonstrations, or educational finance examples.
---

# Quant Research Builder

Builds complete quantitative finance research projects across 10 types. Each includes Python code, backtest results, and HTML presentations.

## 10 Project Types

1. **Factor Model** - Fama-French, momentum, value, quality factors
2. **Portfolio Optimization** - Markowitz, Black-Litterman, risk parity
3. **Risk Analytics** - VaR, CVaR, stress testing, tail risk
4. **Market Microstructure** - Order book, liquidity, execution
5. **Options Pricing** - Greeks, vol surfaces, exotic options
6. **Algorithmic Trading** - Momentum, mean reversion, pairs
7. **Time Series Forecasting** - ARIMA, GARCH, LSTM prices
8. **Volatility Modeling** - Implied/realized vol, forecasting
9. **High-Frequency Analysis** - Tick data, market making
10. **Quantamental** - Fundamental + technical signals

## Workflow

### 1. Initialize
- Select type (1-10)
- Choose strategy, instruments, period creatively
- Document config

### 2. Data & Planning
Use `scripts/data_loader.py`:
```python
from data_loader import (
    simulate_market_data,
    generate_factor_data,
    simulate_order_book,
    generate_options_data
)
prices = simulate_market_data(days=252, frequency='1D')
```

Review `references/quant_patterns.md` for:
- Backtest framework
- Strategy templates
- Risk metrics

### 3. Implement Strategy
Follow patterns from `references/quant_patterns.md`:
- Imports & setup
- Data loading
- Strategy definition (signals, rules)
- Backtest execution
- Performance analysis
- Visualization
- Export metrics.json

**Critical:** No look-ahead bias, include transaction costs

### 4. Visualize
Use `scripts/viz_utils.py`:
```python
from viz_utils import (
    plot_returns_analysis,
    plot_factor_exposures,
    plot_efficient_frontier,
    plot_order_book
)
plot_returns_analysis(returns, 'results/returns.png')
```

Create type-specific plots (see references)

### 5. HTML Presentation
Populate `assets/presentation_template.html` with:
- Strategy name, thesis
- Backtest design, assumptions
- Results with embedded images (base64)
- Risk analysis (drawdown, VaR)
- Economic intuition
- Code with syntax highlighting
- Conclusions

### 6. Output
```
/mnt/user-data/outputs/quant-[type]-[slug]/
├── index.html
├── code/
│   ├── strategy.py
│   └── requirements.txt
├── results/
│   ├── figure_*.png
│   ├── metrics.json
│   └── trades.csv
└── README.md
```

## Creative Freedom

**Agent chooses:**
- Strategy parameters (lookback, thresholds)
- Asset universe
- Risk management rules
- Benchmark selection
- Visualization emphasis

**Agent must:**
- Follow 6-step workflow
- No look-ahead bias
- Include transaction costs
- Run in < 20 minutes
- Generate complete HTML

## Resources

**scripts/**
- `data_loader.py` - Simulate markets, factors, order books, options
- `viz_utils.py` - Returns analysis, correlations, factor exposures, efficient frontier, Greeks, order books

**references/**
- `project_ideas.md` - 75+ quant project ideas
- `quant_patterns.md` - Backtest framework, strategies, risk metrics, portfolio optimization

**assets/**
- `presentation_template.html` - Professional HTML template

## Quality Checklist
- [ ] Code executes (<20 min)
- [ ] No look-ahead bias
- [ ] Transaction costs included
- [ ] 3-4 visualizations
- [ ] Risk-adjusted metrics
- [ ] Complete HTML
- [ ] Economic intuition clear

## Common Pitfalls to Avoid
1. Look-ahead bias
2. Survivorship bias
3. Ignoring transaction costs
4. Overfitting
5. Regime dependence

See full workflow and type-specific guidelines in references.
