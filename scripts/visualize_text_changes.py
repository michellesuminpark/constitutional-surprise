"""
Visual 1: Before/After Text Comparison
Shows examples of abstracts with BIG vs SMALL perplexity changes
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import textwrap

def create_text_comparison_visual():
    """Create side-by-side comparison of original vs manipulated abstracts."""

    # Load data
    df = pd.read_csv('data/macss_full_with_perplexity.csv')

    # Find extreme examples
    biggest_increase = df.nlargest(1, 'perplexity_change').iloc[0]
    smallest_change = df.nsmallest(1, 'perplexity_change').iloc[0]

    print("="*80)
    print("SELECTED EXAMPLES:")
    print("="*80)
    print(f"\nBIGGEST INCREASE:")
    print(f"  Paper: {biggest_increase['paper_id']}")
    print(f"  Original PPL: {biggest_increase['perplexity_original']:.2f}")
    print(f"  Manipulated PPL: {biggest_increase['perplexity_manipulated']:.2f}")
    print(f"  Change: +{biggest_increase['perplexity_change']:.2f}")

    print(f"\nSMALLEST CHANGE:")
    print(f"  Paper: {smallest_change['paper_id']}")
    print(f"  Original PPL: {smallest_change['perplexity_original']:.2f}")
    print(f"  Manipulated PPL: {smallest_change['perplexity_manipulated']:.2f}")
    print(f"  Change: {smallest_change['perplexity_change']:.2f}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('The Perplexity Paradox: When AI "Improves" Academic Writing',
                 fontsize=18, fontweight='bold', y=0.98)

    # --- ROW 1: BIGGEST INCREASE ---
    ax_orig_1 = axes[0, 0]
    ax_manip_1 = axes[0, 1]

    # Original (left)
    orig_text_1 = str(biggest_increase['abstract'])[:600]
    wrapped_orig_1 = textwrap.fill(orig_text_1, width=70)

    ax_orig_1.text(0.05, 0.95, 'ORIGINAL ABSTRACT',
                   fontsize=12, fontweight='bold',
                   transform=ax_orig_1.transAxes, va='top')
    ax_orig_1.text(0.05, 0.88, f"Perplexity: {biggest_increase['perplexity_original']:.1f}",
                   fontsize=11, color='green', fontweight='bold',
                   transform=ax_orig_1.transAxes, va='top')
    ax_orig_1.text(0.05, 0.80, wrapped_orig_1,
                   fontsize=9, transform=ax_orig_1.transAxes, va='top',
                   wrap=True, family='serif')
    ax_orig_1.axis('off')

    # Add background
    rect_orig_1 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                 boxstyle="round,pad=0.01",
                                 transform=ax_orig_1.transAxes,
                                 facecolor='#e8f5e9', edgecolor='green',
                                 linewidth=2, zorder=-1)
    ax_orig_1.add_patch(rect_orig_1)

    # Manipulated (right)
    manip_text_1 = str(biggest_increase['abstract_manipulated'])[:600]
    wrapped_manip_1 = textwrap.fill(manip_text_1, width=70)

    ax_manip_1.text(0.05, 0.95, 'AI-MANIPULATED VERSION',
                    fontsize=12, fontweight='bold',
                    transform=ax_manip_1.transAxes, va='top')
    ax_manip_1.text(0.05, 0.88, f"Perplexity: {biggest_increase['perplexity_manipulated']:.1f}",
                    fontsize=11, color='red', fontweight='bold',
                    transform=ax_manip_1.transAxes, va='top')
    ax_manip_1.text(0.05, 0.80, wrapped_manip_1,
                    fontsize=9, transform=ax_manip_1.transAxes, va='top',
                    wrap=True, family='serif')
    ax_manip_1.axis('off')

    # Add background
    rect_manip_1 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                  boxstyle="round,pad=0.01",
                                  transform=ax_manip_1.transAxes,
                                  facecolor='#ffebee', edgecolor='red',
                                  linewidth=2, zorder=-1)
    ax_manip_1.add_patch(rect_manip_1)

    # Add arrow showing increase
    fig.text(0.5, 0.52, f'↑ +{biggest_increase["perplexity_change"]:.1f} points',
             ha='center', fontsize=14, fontweight='bold', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # --- ROW 2: SMALLEST CHANGE ---
    ax_orig_2 = axes[1, 0]
    ax_manip_2 = axes[1, 1]

    # Original (left)
    orig_text_2 = str(smallest_change['abstract'])[:600]
    wrapped_orig_2 = textwrap.fill(orig_text_2, width=70)

    ax_orig_2.text(0.05, 0.95, 'ORIGINAL ABSTRACT',
                   fontsize=12, fontweight='bold',
                   transform=ax_orig_2.transAxes, va='top')
    ax_orig_2.text(0.05, 0.88, f"Perplexity: {smallest_change['perplexity_original']:.1f}",
                   fontsize=11, color='green', fontweight='bold',
                   transform=ax_orig_2.transAxes, va='top')
    ax_orig_2.text(0.05, 0.80, wrapped_orig_2,
                   fontsize=9, transform=ax_orig_2.transAxes, va='top',
                   wrap=True, family='serif')
    ax_orig_2.axis('off')

    # Add background
    rect_orig_2 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                 boxstyle="round,pad=0.01",
                                 transform=ax_orig_2.transAxes,
                                 facecolor='#e8f5e9', edgecolor='green',
                                 linewidth=2, zorder=-1)
    ax_orig_2.add_patch(rect_orig_2)

    # Manipulated (right)
    manip_text_2 = str(smallest_change['abstract_manipulated'])[:600]
    wrapped_manip_2 = textwrap.fill(manip_text_2, width=70)

    ax_manip_2.text(0.05, 0.95, 'AI-MANIPULATED VERSION',
                    fontsize=12, fontweight='bold',
                    transform=ax_manip_2.transAxes, va='top')
    ax_manip_2.text(0.05, 0.88, f"Perplexity: {smallest_change['perplexity_manipulated']:.1f}",
                    fontsize=11, color='blue', fontweight='bold',
                    transform=ax_manip_2.transAxes, va='top')
    ax_manip_2.text(0.05, 0.80, wrapped_manip_2,
                    fontsize=9, transform=ax_manip_2.transAxes, va='top',
                    wrap=True, family='serif')
    ax_manip_2.axis('off')

    # Add background
    rect_manip_2 = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                  boxstyle="round,pad=0.01",
                                  transform=ax_manip_2.transAxes,
                                  facecolor='#e3f2fd', edgecolor='blue',
                                  linewidth=2, zorder=-1)
    ax_manip_2.add_patch(rect_manip_2)

    # Add arrow showing minimal change
    fig.text(0.5, 0.02, f'↓ {smallest_change["perplexity_change"]:.1f} points (minimal change)',
             ha='center', fontsize=14, fontweight='bold', color='blue',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save
    output_path = 'visualizations/visual_01_text_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved to {output_path}")
    plt.close()

    # Also create a simple table version for blog
    create_table_version(biggest_increase, smallest_change)

def create_table_version(big, small):
    """Create a cleaner table-style comparison."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Title
    fig.suptitle('Can AI "Game" the Perplexity Metric?\nComparing High vs Low Impact Manipulations',
                 fontsize=16, fontweight='bold', y=0.98)

    # Create table data
    table_data = []

    # Header
    table_data.append(['', 'EXAMPLE 1: BIG CHANGE', 'EXAMPLE 2: MINIMAL CHANGE'])

    # Perplexity change
    table_data.append([
        '📊 Perplexity Change',
        f'+{big["perplexity_change"]:.1f} points\n({big["perplexity_original"]:.1f} → {big["perplexity_manipulated"]:.1f})',
        f'{small["perplexity_change"]:.1f} points\n({small["perplexity_original"]:.1f} → {small["perplexity_manipulated"]:.1f})'
    ])

    # Original text
    orig_big = textwrap.fill(str(big['abstract'])[:200] + '...', width=40)
    orig_small = textwrap.fill(str(small['abstract'])[:200] + '...', width=40)
    table_data.append(['📄 Original', orig_big, orig_small])

    # Manipulated text
    manip_big = textwrap.fill(str(big['abstract_manipulated'])[:200] + '...', width=40)
    manip_small = textwrap.fill(str(small['abstract_manipulated'])[:200] + '...', width=40)
    table_data.append(['🤖 Manipulated', manip_big, manip_small])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.2, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 4)

    # Style header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=10)

    # Style metric row
    table[(1, 1)].set_facecolor('#ffcccc')
    table[(1, 2)].set_facecolor('#cce5ff')

    # Style left column
    for i in range(1, 4):
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 0)].set_text_props(weight='bold')

    ax.axis('off')

    # Save
    output_path = 'visualizations/visual_01_table_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)

    print("Creating before/after text comparisons...")
    create_text_comparison_visual()
    print("\n✅ Done! Created 2 versions of Visual 1")
