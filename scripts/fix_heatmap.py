"""
Quick fix for Visual 4: Feature Heatmap
Normalizes the data and uses better colormap to show patterns clearly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
print("Loading data...")
df = pd.read_csv('data/macss_full_with_perplexity.csv')

# Load the SAE features
print("Loading SAE features...")
hidden_features = np.load('results/sae_features.npy')
print(f"Loaded SAE features: {hidden_features.shape}")

# SAE was trained on 100 sampled abstracts (200 total: 100 orig + 100 manip)
# Split into original and manipulated halves
n_samples = len(hidden_features) // 2
hidden_orig = hidden_features[:n_samples]
hidden_manip = hidden_features[n_samples:]

# Sample the same number from the full dataset for matching
# We'll use a random sample since the SAE used random sampling
np.random.seed(42)  # Same seed as SAE script
df_sample = df.sample(n=n_samples, random_state=42)
perplexity_change = df_sample['perplexity_change'].values

print(f"Using {n_samples} matched samples")

print(f"Using {len(hidden_manip)} samples, {hidden_manip.shape[1]} features")

# Find top features by correlation (skip zero-variance features)
print("\nFinding top features...")
correlations = []
valid_features = []

for i in range(hidden_manip.shape[1]):
    # Skip features with zero or near-zero variance
    if np.std(hidden_manip[:, i]) > 0.01:
        corr = np.corrcoef(hidden_manip[:, i], perplexity_change)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))
            valid_features.append(i)

print(f"Found {len(valid_features)} features with non-zero variance")

# Get top 10 features
if len(valid_features) < 10:
    print(f"Warning: Only {len(valid_features)} valid features found")
    top_indices = range(len(valid_features))
else:
    top_indices = np.argsort(correlations)[-10:][::-1]

top_features = [valid_features[i] for i in top_indices]
top_correlations = [correlations[i] for i in top_indices]

print(f"Top 3 feature correlations: {top_correlations[:3]}")

# Select samples to display
n_samples = min(50, len(hidden_manip))
indices = np.random.choice(len(hidden_manip), n_samples, replace=False)
indices = np.sort(indices)  # Sort for better visualization

# Extract heatmap data
heatmap_data = hidden_manip[indices][:, top_features]

# NORMALIZE: Z-score each feature (row-wise normalization)
print("\nNormalizing features...")
scaler = StandardScaler()
heatmap_data_norm = scaler.fit_transform(heatmap_data.T).T

# Create the improved heatmap with more space for labels
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Use a better colormap - viridis or RdYlBu with better centering
sns.heatmap(heatmap_data_norm.T,
            cmap='RdYlBu_r',
            center=0,
            vmin=-2, vmax=2,  # Clip outliers for better color contrast
            xticklabels=False,
            yticklabels=[f'Feature {i}' for i in top_features],
            cbar_kws={'label': 'Activation Strength (z-score)'},
            linewidths=0.1,
            linecolor='gray')

ax.set_xlabel('Abstracts (randomly sampled)', fontsize=12, fontweight='bold')
ax.set_ylabel('SAE Features', fontsize=12, fontweight='bold')
ax.set_title('Top 10 SAE Features Correlated with Perplexity Change\n(Features ranked by correlation strength)',
             fontsize=14, fontweight='bold', pad=20)

# Add correlation values on the right - make them bigger and clearer
for i, (feat_idx, corr) in enumerate(zip(top_features, top_correlations)):
    # Vary color by correlation strength
    if corr > 0.3:
        color = 'darkgreen'
        alpha = 0.9
    elif corr > 0.25:
        color = 'green'
        alpha = 0.8
    else:
        color = 'darkblue'
        alpha = 0.7

    ax.text(n_samples + 3, i + 0.5, f'r = {corr:.3f}',
            va='center', ha='left', fontsize=11, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=alpha, edgecolor=color, linewidth=2))

# Adjust layout to ensure labels aren't cut off
plt.subplots_adjust(right=0.88)  # Make room for correlation labels
plt.savefig('visualizations/visual_04_feature_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✅ Fixed heatmap saved: visualizations/visual_04_feature_heatmap.png")
plt.close()

print("\nKey improvements:")
print("  - Features are now z-score normalized (shows relative patterns)")
print("  - Color scale adjusted to [-2, 2] for better contrast")
print("  - Top correlations highlighted in green")
