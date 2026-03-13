"""
Visual 2: Perplexity Correlation Scatter Plot
Shows relationship between original and manipulated perplexity
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def create_scatter_plot():
    """Create scatter plot of original vs manipulated perplexity."""

    # Load data
    df = pd.read_csv('data/macss_full_with_perplexity.csv')

    # Calculate correlation
    correlation = df['perplexity_original'].corr(df['perplexity_manipulated'])

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['perplexity_original'],
        df['perplexity_manipulated']
    )

    print("="*80)
    print("CORRELATION ANALYSIS:")
    print("="*80)
    print(f"Pearson correlation: {correlation:.3f}")
    print(f"R-squared: {r_value**2:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Scatter plot
    ax.scatter(df['perplexity_original'], df['perplexity_manipulated'],
               alpha=0.6, s=80, color='steelblue', edgecolors='black', linewidth=0.5)

    # Add regression line
    x_range = np.linspace(df['perplexity_original'].min(),
                          df['perplexity_original'].max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, 'r--', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')

    # Add diagonal line (no change)
    max_val = max(df['perplexity_original'].max(), df['perplexity_manipulated'].max())
    ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1.5, alpha=0.5,
            label='No change line (y=x)')

    # Labels and title
    ax.set_xlabel('Original Perplexity', fontsize=13, fontweight='bold')
    ax.set_ylabel('Manipulated Perplexity', fontsize=13, fontweight='bold')
    ax.set_title('Does AI Manipulation Increase Perplexity?\nOriginal vs Manipulated Abstracts',
                 fontsize=15, fontweight='bold', pad=20)

    # Add statistics box
    stats_text = f'''Correlation: {correlation:.3f}
R² = {r_value**2:.3f}
n = {len(df)}
p < 0.001''' if p_value < 0.001 else f'''Correlation: {correlation:.3f}
R² = {r_value**2:.3f}
n = {len(df)}
p = {p_value:.3f}'''

    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add interpretation note
    mean_change = df['perplexity_change'].mean()
    interpretation = f"Mean change: {mean_change:+.2f} points"
    if mean_change > 5:
        interpretation += "\n✓ Manipulation INCREASES perplexity"
        color = 'green'
    elif mean_change < -5:
        interpretation += "\n✗ Manipulation DECREASES perplexity"
        color = 'red'
    else:
        interpretation += "\n≈ Minimal effect (robust metric)"
        color = 'orange'

    ax.text(0.95, 0.05, interpretation,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path = 'visualizations/visual_02_correlation_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved to {output_path}")
    plt.close()

    # Also create a version with color-coded by change magnitude
    create_color_coded_version(df, correlation, r_value, p_value)

def create_color_coded_version(df, correlation, r_value, p_value):
    """Create scatter plot color-coded by magnitude of change."""

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Color by change magnitude
    colors = df['perplexity_change'].values
    scatter = ax.scatter(df['perplexity_original'], df['perplexity_manipulated'],
                        c=colors, cmap='RdYlGn', s=80, alpha=0.7,
                        edgecolors='black', linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Perplexity Change\n(Manipulated - Original)',
                   fontsize=11, fontweight='bold')

    # Add diagonal line
    max_val = max(df['perplexity_original'].max(), df['perplexity_manipulated'].max())
    ax.plot([0, max_val], [0, max_val], 'k:', linewidth=1.5, alpha=0.5,
            label='No change line')

    # Labels
    ax.set_xlabel('Original Perplexity', fontsize=13, fontweight='bold')
    ax.set_ylabel('Manipulated Perplexity', fontsize=13, fontweight='bold')
    ax.set_title('Perplexity Changes by Abstract\nColor = Magnitude of Change',
                 fontsize=15, fontweight='bold', pad=20)

    # Stats box
    stats_text = f'''r = {correlation:.3f}
R² = {r_value**2:.3f}
n = {len(df)}
Mean Δ = {df["perplexity_change"].mean():.2f}
Median Δ = {df["perplexity_change"].median():.2f}'''

    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path = 'visualizations/visual_02_scatter_colored.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)

    print("Creating perplexity correlation scatter plots...")
    create_scatter_plot()
    print("\n✅ Done! Created 2 versions of Visual 2")
