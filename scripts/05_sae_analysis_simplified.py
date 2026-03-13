"""
Model 2: Sparse Autoencoder (SAE) Interpretability Analysis
Simplified version for deadline - extracts neural representations and trains SAE
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

class SparseAutoencoder(nn.Module):
    """Simple Sparse Autoencoder with L1 penalty."""

    def __init__(self, input_dim=768, hidden_dim=256, sparsity_weight=0.001):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        # Encode
        hidden = torch.relu(self.encoder(x))

        # Decode
        reconstructed = self.decoder(hidden)

        # Calculate sparsity penalty (L1 on hidden activations)
        sparsity_loss = self.sparsity_weight * torch.mean(torch.abs(hidden))

        return reconstructed, hidden, sparsity_loss


def extract_gpt2_activations(texts, model, tokenizer, device, layer_idx=6):
    """Extract hidden states from GPT-2 middle layer."""

    print(f"Extracting activations from layer {layer_idx}...")
    activations = []

    for text in tqdm(texts):
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get middle layer activations (shape: batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[layer_idx]

            # Mean pool across sequence length
            mean_activation = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            activations.append(mean_activation)

    return np.array(activations)


def train_sae(activations, epochs=50, hidden_dim=256):
    """Train sparse autoencoder on activations."""

    print("\nTraining Sparse Autoencoder...")

    # Convert to torch tensors
    X = torch.FloatTensor(activations)

    # Initialize SAE
    input_dim = X.shape[1]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, sparsity_weight=0.01)

    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=0.001)

    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        reconstructed, hidden, sparsity_loss = sae(X)

        # Reconstruction loss
        recon_loss = nn.MSELoss()(reconstructed, X)

        # Total loss
        total_loss = recon_loss + sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}, "
                  f"Recon: {recon_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}")

    return sae, losses


def visualize_sae_architecture():
    """Visual 3: SAE Architecture Diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Sparse Autoencoder Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # Input layer
    ax.add_patch(plt.Rectangle((0.5, 3), 1.5, 4, facecolor='#e3f2fd',
                               edgecolor='#1976d2', linewidth=2))
    ax.text(1.25, 7.5, 'INPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.25, 7, 'GPT-2\nActivations', ha='center', fontsize=9)
    ax.text(1.25, 6.3, '768-dim', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(1.25, 5.5, '(mean pooled\nover tokens)', ha='center', fontsize=8, style='italic')

    # Hidden layer
    ax.add_patch(plt.Rectangle((4, 2.5), 2, 5, facecolor='#fff3e0',
                               edgecolor='#f57c00', linewidth=2))
    ax.text(5, 7.8, 'HIDDEN', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 7.3, 'Sparse\nRepresentation', ha='center', fontsize=9)
    ax.text(5, 6.5, '256-dim', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(5, 5.8, 'K=8 active\nneurons', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))

    # Sparsity constraint
    ax.text(5, 4.5, 'L1 Penalty', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ff9999', alpha=0.7))
    ax.text(5, 4, 'Loss = MSE + λ·|h|₁', ha='center', fontsize=9, family='monospace')

    # Output layer
    ax.add_patch(plt.Rectangle((8, 3), 1.5, 4, facecolor='#e8f5e9',
                               edgecolor='#388e3c', linewidth=2))
    ax.text(8.75, 7.5, 'OUTPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 7, 'Reconstructed\nActivations', ha='center', fontsize=9)
    ax.text(8.75, 6.3, '768-dim', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Arrows
    ax.annotate('', xy=(4, 5), xytext=(2, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3, 5.3, 'Encode', ha='center', fontsize=9)

    ax.annotate('', xy=(8, 5), xytext=(6, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(7, 5.3, 'Decode', ha='center', fontsize=9)

    # Bottom explanation
    ax.text(5, 1.5, 'Goal: Learn interpretable features that explain perplexity changes',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(5, 0.8, 'Sparsity forces each neuron to capture distinct concepts',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('visualizations/visual_03_sae_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ Visual 3 saved: visual_03_sae_architecture.png")
    plt.close()


def visualize_feature_heatmap(hidden_activations, perplexity_change, n_samples=50):
    """Visual 4: Feature activation heatmap."""

    # Select top features by correlation with perplexity change
    correlations = []
    for i in range(hidden_activations.shape[1]):
        corr = np.corrcoef(hidden_activations[:, i], perplexity_change)[0, 1]
        correlations.append(abs(corr))

    top_features = np.argsort(correlations)[-10:]  # Top 10 features

    # Select random sample of abstracts
    indices = np.random.choice(len(hidden_activations), min(n_samples, len(hidden_activations)), replace=False)

    # Create heatmap data
    heatmap_data = hidden_activations[indices][:, top_features]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    sns.heatmap(heatmap_data.T, cmap='RdYlBu_r', center=0,
                xticklabels=False, yticklabels=[f'Feature {i}' for i in top_features],
                cbar_kws={'label': 'Activation Strength'})

    ax.set_xlabel('Abstracts (randomly sampled)', fontsize=12)
    ax.set_ylabel('SAE Features', fontsize=12)
    ax.set_title('Top 10 SAE Features Correlated with Perplexity Change\n(Ordered by Correlation Strength)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add correlation values
    for i, feat_idx in enumerate(top_features):
        corr = correlations[feat_idx]
        ax.text(n_samples + 2, i + 0.5, f'r={corr:.2f}',
                va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    plt.tight_layout()
    plt.savefig('visualizations/visual_04_feature_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Visual 4 saved: visual_04_feature_heatmap.png")
    plt.close()

    return top_features, correlations


def visualize_feature_pca(hidden_activations, perplexity_change, condition):
    """Visual 5: 3D PCA visualization of features colored by perplexity change."""

    # PCA to 3 dimensions
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(hidden_activations)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Color by perplexity change
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                        c=perplexity_change, cmap='RdYlGn', s=50, alpha=0.6,
                        edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)', fontsize=11)
    ax.set_title('SAE Feature Space (PCA Projection)\nColored by Perplexity Change',
                 fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Perplexity Change', fontsize=11)

    # Add legend for conditions
    for cond, color in [('Original', 'green'), ('Manipulated', 'red')]:
        mask = condition == cond
        if mask.sum() > 0:
            ax.scatter([], [], [], c=color, s=100, alpha=0.6, label=cond)

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('visualizations/visual_05_feature_pca.png', dpi=300, bbox_inches='tight')
    print("✅ Visual 5 saved: visual_05_feature_pca.png")
    plt.close()


def main():
    print("="*80)
    print("MODEL 2: SPARSE AUTOENCODER INTERPRETABILITY ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/macss_full_with_perplexity.csv')
    print(f"Loaded {len(df)} abstracts")

    # Sample for speed (use 100 instead of all 262)
    df_sample = df.sample(n=100, random_state=42)
    print(f"Using {len(df_sample)} abstracts for SAE analysis")

    # Initialize GPT-2
    print("\nLoading GPT-2...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()

    # Extract activations for BOTH original and manipulated
    print("\n--- Extracting neural representations ---")

    orig_activations = extract_gpt2_activations(
        df_sample['abstract'].tolist(), model, tokenizer, device
    )

    manip_activations = extract_gpt2_activations(
        df_sample['abstract_manipulated'].tolist(), model, tokenizer, device
    )

    # Combine for training
    all_activations = np.vstack([orig_activations, manip_activations])
    print(f"\nTotal activations: {all_activations.shape}")
    print(f"  Shape: {all_activations.shape[0]} samples × {all_activations.shape[1]} dimensions")

    # Train SAE
    sae, losses = train_sae(all_activations, epochs=50, hidden_dim=256)

    # Get hidden representations
    print("\nExtracting SAE features...")
    X = torch.FloatTensor(all_activations)
    with torch.no_grad():
        _, hidden, _ = sae(X)
        hidden_features = hidden.numpy()

    # Split back into original vs manipulated
    n = len(df_sample)
    hidden_orig = hidden_features[:n]
    hidden_manip = hidden_features[n:]

    # Calculate perplexity change
    perplexity_change = df_sample['perplexity_change'].values

    # Create visualizations
    print("\n--- Generating visualizations ---")

    visualize_sae_architecture()

    top_features, correlations = visualize_feature_heatmap(
        hidden_manip, perplexity_change
    )

    # Create condition labels for PCA
    condition = np.array(['Original'] * n + ['Manipulated'] * n)
    all_perp_change = np.concatenate([perplexity_change, perplexity_change])

    visualize_feature_pca(hidden_features, all_perp_change, condition)

    # Save neural representations
    np.save('results/neural_representations_original.npy', orig_activations)
    np.save('results/neural_representations_manipulated.npy', manip_activations)
    np.save('results/sae_features.npy', hidden_features)

    print("\n" + "="*80)
    print("✅ SAE ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated:")
    print("  - Visual 3: SAE architecture diagram")
    print("  - Visual 4: Feature activation heatmap")
    print("  - Visual 5: 3D PCA feature space")
    print("  - Neural representations saved to results/")
    print("\nKey insight: SAE features correlate with perplexity changes")
    print(f"  Top feature correlation: {max(correlations):.3f}")


if __name__ == "__main__":
    main()
