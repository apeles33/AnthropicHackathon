#!/usr/bin/env python3
"""
Trading Analytics Project Generator

This script orchestrates the generation of complete trading/quant finance projects.
It handles project type selection, code generation, backtesting, visualization,
and HTML output creation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
import json

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ProjectGenerator:
    """Main class for generating trading analytics projects"""
    
    PROJECT_TYPES = {
        1: "Momentum Bot",
        2: "Mean Reversion Strategy",
        3: "Sentiment Analyzer",
        4: "Options Pricer",
        5: "Portfolio Optimizer",
        6: "Market Maker Simulator",
        7: "Arbitrage Scanner",
        8: "Risk Dashboard",
        9: "Technical Indicators Library",
        10: "Backtest Engine",
        11: "Market Microstructure Analyzer",
        12: "Pairs Trading Strategy",
        13: "Volatility Forecaster",
        14: "High-Frequency Trading Simulator",
        15: "Multi-Asset Allocator"
    }
    
    def __init__(self, project_type: int):
        self.project_type = project_type
        self.project_name = self.PROJECT_TYPES.get(project_type, "Unknown Project")
        self.results = {}
        self.code_snippets = []
        self.visualizations = []
        
    def generate_data(self) -> pd.DataFrame:
        """Generate appropriate data for the project type"""
        # This would be implemented with data_generator.py functions
        pass
    
    def implement_algorithm(self):
        """Implement the core trading algorithm"""
        # Project-specific implementation
        pass
    
    def run_backtest(self) -> Dict:
        """Run the backtest and collect metrics"""
        # Calculate performance metrics
        metrics = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }
        return metrics
    
    def create_visualizations(self):
        """Generate all charts for the project"""
        # Create equity curve, drawdown, returns distribution, etc.
        pass
    
    def plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def generate_html_output(self, template_path: str, output_path: str):
        """Create the final HTML presentation"""
        # Load template and fill with project data
        pass
    
    def execute(self, output_dir: str = "."):
        """Main execution flow"""
        print(f"Generating {self.project_name}...")
        
        # 1. Generate/load data
        print("  - Generating data...")
        data = self.generate_data()
        
        # 2. Implement algorithm
        print("  - Implementing algorithm...")
        self.implement_algorithm()
        
        # 3. Run backtest
        print("  - Running backtest...")
        self.results = self.run_backtest()
        
        # 4. Create visualizations
        print("  - Creating visualizations...")
        self.create_visualizations()
        
        # 5. Generate HTML
        print("  - Generating HTML output...")
        output_path = f"{output_dir}/{self.project_name.replace(' ', '_')}.html"
        self.generate_html_output("../assets/project_template.html", output_path)
        
        print(f"✓ Project complete: {output_path}")
        return output_path


def calculate_metrics(returns: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calculate standard performance metrics for a strategy.
    
    Args:
        returns: Series of strategy returns
        trades: Optional DataFrame with trade details
        
    Returns:
        Dictionary of performance metrics
    """
    # Returns metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Risk-adjusted metrics
    sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Trade metrics
    if trades is not None:
        num_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if len(trades[trades['pnl'] < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        num_trades = 0
        win_rate = 0
        profit_factor = 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def create_equity_curve_chart(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> str:
    """Create equity curve visualization and return as base64"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategy_equity = (1 + returns).cumprod()
    ax.plot(strategy_equity.index, strategy_equity.values, label='Strategy', linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_equity = (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_equity.index, benchmark_equity.values, 
                label='Benchmark', linewidth=2, alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_drawdown_chart(returns: pd.Series) -> str:
    """Create drawdown visualization and return as base64"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown Over Time')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_returns_distribution(returns: pd.Series) -> str:
    """Create returns distribution visualization and return as base64"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(returns.values, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    ax1.set_xlabel('Daily Returns')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Returns Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns.values, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)')
    ax2.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_project.py <project_type>")
        print("\nAvailable project types:")
        for num, name in ProjectGenerator.PROJECT_TYPES.items():
            print(f"  {num}: {name}")
        sys.exit(1)
    
    project_type = int(sys.argv[1])
    generator = ProjectGenerator(project_type)
    generator.execute()
