"""
Visualization Utilities for Research Projects
Standard plotting functions for ML and Quant projects
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# ===== ML Visualizations =====

def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training and validation loss/accuracy
    
    Args:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: where to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train', linewidth=2)
        if 'val_acc' in history:
            ax2.plot(history['val_acc'], label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_attention_heatmap(attention_weights, tokens, save_path='attention.png'):
    """
    Visualize attention weights
    
    Args:
        attention_weights: (seq_len, seq_len) array
        tokens: list of token strings
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, 
                cmap='viridis', cbar=True)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmap to {save_path}")

def plot_embeddings_2d(embeddings, labels, save_path='embeddings.png'):
    """
    Plot 2D embeddings (from t-SNE or UMAP)
    
    Args:
        embeddings: (n_samples, 2) array
        labels: (n_samples,) array of class labels
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Embedding Visualization')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved embeddings to {save_path}")

def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {save_path}")

# ===== Quant Visualizations =====

def plot_returns_analysis(returns, benchmark_returns=None, save_path='returns_analysis.png'):
    """
    Plot cumulative returns and drawdown
    
    Args:
        returns: pandas Series of returns
        benchmark_returns: optional benchmark returns
        save_path: where to save
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    ax1.plot(cum_returns.index, cum_returns.values, linewidth=2, label='Strategy')
    
    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        ax1.plot(cum_bench.index, cum_bench.values, linewidth=2, 
                label='Benchmark', alpha=0.7)
    
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title('Cumulative Returns Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved returns analysis to {save_path}")

def plot_correlation_heatmap(dataframe, save_path='correlation_heatmap.png'):
    """Plot correlation matrix as heatmap"""
    corr = dataframe.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")

def plot_factor_exposures(exposures_dict, save_path='factor_exposures.png'):
    """
    Plot factor exposures as bar chart
    
    Args:
        exposures_dict: dict mapping factor names to exposure values
    """
    factors = list(exposures_dict.keys())
    values = list(exposures_dict.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.barh(factors, values, color=colors, alpha=0.7)
    plt.xlabel('Exposure')
    plt.title('Factor Exposures')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved factor exposures to {save_path}")

def plot_efficient_frontier(returns, risks, sharpe_ratios, save_path='efficient_frontier.png'):
    """
    Plot efficient frontier
    
    Args:
        returns: array of portfolio returns
        risks: array of portfolio volatilities
        sharpe_ratios: array of Sharpe ratios
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(risks, returns, c=sharpe_ratios, cmap='viridis', s=20)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved efficient frontier to {save_path}")

def plot_volatility_cone(realized_vols, save_path='volatility_cone.png'):
    """
    Plot volatility cone showing percentiles
    
    Args:
        realized_vols: DataFrame with columns for different horizons
    """
    percentiles = [10, 25, 50, 75, 90]
    horizons = realized_vols.columns
    
    plt.figure(figsize=(10, 6))
    
    for p in percentiles:
        values = realized_vols.quantile(p/100)
        plt.plot(horizons, values, marker='o', label=f'{p}th percentile', linewidth=2)
    
    plt.xlabel('Horizon (days)')
    plt.ylabel('Realized Volatility')
    plt.title('Volatility Cone')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved volatility cone to {save_path}")

def plot_greeks_surface(strikes, ttms, greeks_matrix, greek_name='Delta', 
                       save_path='greeks_surface.png'):
    """
    Plot Greeks as 3D surface
    
    Args:
        strikes: array of strike prices
        ttms: array of times to maturity
        greeks_matrix: 2D array of Greek values
        greek_name: name of Greek being plotted
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(strikes, ttms)
    surf = ax.plot_surface(X, Y, greeks_matrix, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity (days)')
    ax.set_zlabel(greek_name)
    ax.set_title(f'{greek_name} Surface')
    
    fig.colorbar(surf, shrink=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {greek_name} surface to {save_path}")

def plot_order_book(bids, asks, save_path='order_book.png'):
    """
    Plot order book depth
    
    Args:
        bids: list of {'price': p, 'size': s} dicts
        asks: list of {'price': p, 'size': s} dicts
    """
    bid_prices = [b['price'] for b in bids]
    bid_sizes = [b['size'] for b in bids]
    ask_prices = [a['price'] for a in asks]
    ask_sizes = [a['size'] for a in asks]
    
    plt.figure(figsize=(12, 6))
    
    # Cumulative sizes
    bid_cumsum = np.cumsum(bid_sizes[::-1])[::-1]
    ask_cumsum = np.cumsum(ask_sizes)
    
    plt.fill_between(bid_prices, 0, bid_cumsum, alpha=0.3, color='green', label='Bids')
    plt.fill_between(ask_prices, 0, ask_cumsum, alpha=0.3, color='red', label='Asks')
    
    plt.xlabel('Price')
    plt.ylabel('Cumulative Size')
    plt.title('Order Book Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved order book to {save_path}")

def plot_risk_return_scatter(returns, risks, labels, save_path='risk_return.png'):
    """
    Plot risk-return scatter for multiple assets/strategies
    
    Args:
        returns: array of returns
        risks: array of risks (volatilities)
        labels: list of asset/strategy names
    """
    plt.figure(figsize=(10, 6))
    
    for i, label in enumerate(labels):
        plt.scatter(risks[i], returns[i], s=100, label=label, alpha=0.7)
        plt.annotate(label, (risks[i], returns[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    plt.xlabel('Risk (Volatility)')
    plt.ylabel('Return')
    plt.title('Risk-Return Tradeoff')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved risk-return scatter to {save_path}")

# ===== Utility Functions =====

def save_figure_grid(images, titles, save_path='figure_grid.png', rows=2):
    """
    Save multiple images in a grid
    
    Args:
        images: list of numpy arrays (images)
        titles: list of titles for each image
        save_path: where to save
        rows: number of rows in grid
    """
    n = len(images)
    cols = int(np.ceil(n / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved figure grid to {save_path}")

def create_comparison_plot(results_dict, metric_name, save_path='comparison.png'):
    """
    Create bar plot comparing multiple methods
    
    Args:
        results_dict: {method_name: metric_value}
        metric_name: name of metric being compared
    """
    methods = list(results_dict.keys())
    values = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values, alpha=0.7, edgecolor='black')
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (method, value) in enumerate(zip(methods, values)):
        plt.text(i, value, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")
