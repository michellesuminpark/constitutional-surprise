"""
Visual 13: Full Pipeline Infographic
Shows the complete project workflow from data to insights
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_pipeline_visual():
    """Create project pipeline infographic."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    fig.suptitle('Constitutional Surprise: Project Pipeline',
                 fontsize=18, fontweight='bold', y=0.98)

    # STAGE 1: DATA
    stage1_box = FancyBboxPatch((0.5, 8), 2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#e3f2fd', edgecolor='#1976d2',
                                linewidth=3)
    ax.add_patch(stage1_box)
    ax.text(1.5, 8.8, 'STAGE 1: DATA', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 8.5, '270 MACSS\nAbstracts', ha='center', fontsize=9)

    # STAGE 2: MANIPULATION
    stage2_box = FancyBboxPatch((3.5, 8), 2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#fff3e0', edgecolor='#f57c00',
                                linewidth=3)
    ax.add_patch(stage2_box)
    ax.text(4.5, 8.8, 'STAGE 2: MANIPULATION', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.5, 8.5, 'GPT-3.5\n(Jargon Inflation)', ha='center', fontsize=9)

    # STAGE 3: MEASUREMENT
    stage3_box = FancyBboxPatch((6.5, 8), 2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#f3e5f5', edgecolor='#7b1fa2',
                                linewidth=3)
    ax.add_patch(stage3_box)
    ax.text(7.5, 8.8, 'STAGE 3: MEASUREMENT', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 8.5, 'GPT-2\n(Perplexity)', ha='center', fontsize=9)

    # Arrows between stages 1-2-3
    arrow1 = FancyArrowPatch((2.5, 8.6), (3.5, 8.6),
                            arrowstyle='->', mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow1)

    arrow2 = FancyArrowPatch((5.5, 8.6), (6.5, 8.6),
                            arrowstyle='->', mutation_scale=30, linewidth=2, color='black')
    ax.add_patch(arrow2)

    # MODEL BOXES (arranged vertically)
    models_x = 1
    models_y_start = 6
    model_height = 1

    # Model 1: DML
    model1_box = FancyBboxPatch((models_x, models_y_start), 3.5, model_height,
                                boxstyle="round,pad=0.08",
                                facecolor='#c8e6c9', edgecolor='#388e3c',
                                linewidth=2)
    ax.add_patch(model1_box)
    ax.text(models_x + 0.2, models_y_start + 0.7, 'MODEL 1: Causal Inference (DML)',
            fontsize=10, fontweight='bold')
    ax.text(models_x + 0.2, models_y_start + 0.4,
            '→ Does manipulation CAUSE perplexity change?',
            fontsize=8)
    ax.text(models_x + 0.2, models_y_start + 0.1,
            'Output: Treatment effect estimate + CI',
            fontsize=7, style='italic')

    # Model 2: SAE
    model2_box = FancyBboxPatch((models_x, models_y_start - 1.3), 3.5, model_height,
                                boxstyle="round,pad=0.08",
                                facecolor='#ffccbc', edgecolor='#d84315',
                                linewidth=2)
    ax.add_patch(model2_box)
    ax.text(models_x + 0.2, models_y_start - 1.3 + 0.7, 'MODEL 2: Interpretability (SAE)',
            fontsize=10, fontweight='bold')
    ax.text(models_x + 0.2, models_y_start - 1.3 + 0.4,
            '→ HOW does it work? (syntax vs semantics)',
            fontsize=8)
    ax.text(models_x + 0.2, models_y_start - 1.3 + 0.1,
            'Output: Monosemantic features (F_1768)',
            fontsize=7, style='italic')

    # Model 3: Multi-Agent
    model3_box = FancyBboxPatch((models_x, models_y_start - 2.6), 3.5, model_height,
                                boxstyle="round,pad=0.08",
                                facecolor='#b3e5fc', edgecolor='#0277bd',
                                linewidth=2)
    ax.add_patch(model3_box)
    ax.text(models_x + 0.2, models_y_start - 2.6 + 0.7, 'MODEL 3: Multi-Agent Debate',
            fontsize=10, fontweight='bold')
    ax.text(models_x + 0.2, models_y_start - 2.6 + 0.4,
            '→ Can collaboration improve quality?',
            fontsize=8)
    ax.text(models_x + 0.2, models_y_start - 2.6 + 0.1,
            'Output: Debated abstracts + provenance',
            fontsize=7, style='italic')

    # Model 4: RL
    model4_box = FancyBboxPatch((models_x, models_y_start - 3.9), 3.5, model_height,
                                boxstyle="round,pad=0.08",
                                facecolor='#f8bbd0', edgecolor='#c2185b',
                                linewidth=2)
    ax.add_patch(model4_box)
    ax.text(models_x + 0.2, models_y_start - 3.9 + 0.7, 'MODEL 4: Constitutional RL',
            fontsize=10, fontweight='bold')
    ax.text(models_x + 0.2, models_y_start - 3.9 + 0.4,
            '→ Train agent with gaming penalty',
            fontsize=8)
    ax.text(models_x + 0.2, models_y_start - 3.9 + 0.1,
            'Output: R = -PPL + λ·F_1768',
            fontsize=7, style='italic')

    # Arrows from Stage 3 to Models
    for i, y_offset in enumerate([0, -1.3, -2.6, -3.9]):
        arrow = FancyArrowPatch((7.5, 7.5), (models_x + 3.5, models_y_start + y_offset + 0.5),
                               arrowstyle='->', mutation_scale=20, linewidth=1.5,
                               color='gray', linestyle='--', alpha=0.6)
        ax.add_patch(arrow)

    # VALIDATION BOX (right side)
    valid_box = FancyBboxPatch((5.5, 3.5), 3.5, 2,
                               boxstyle="round,pad=0.1",
                               facecolor='#fff9c4', edgecolor='#f57f17',
                               linewidth=3)
    ax.add_patch(valid_box)
    ax.text(7.25, 5.1, 'VALIDATION', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.25, 4.7, 'Human Ratings (n=30)', ha='center', fontsize=9)
    ax.text(7.25, 4.3, '• Novelty', ha='center', fontsize=8)
    ax.text(7.25, 4.0, '• Clarity', ha='center', fontsize=8)
    ax.text(7.25, 3.7, '• Substance', ha='center', fontsize=8)

    # Arrows from models to validation
    for i, y_offset in enumerate([0, -1.3, -2.6, -3.9]):
        if i in [1, 2, 3]:  # Models 2, 3, 4 feed into validation
            arrow = FancyArrowPatch((models_x + 3.5, models_y_start + y_offset + 0.5),
                                   (5.5, 4.5),
                                   arrowstyle='->', mutation_scale=20, linewidth=1.5,
                                   color='gray', linestyle='--', alpha=0.6)
            ax.add_patch(arrow)

    # KEY FINDINGS BOX (bottom)
    findings_box = FancyBboxPatch((1, 0.5), 7.5, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#e8f5e9', edgecolor='#2e7d32',
                                  linewidth=3)
    ax.add_patch(findings_box)
    ax.text(4.75, 1.7, '🎯 KEY FINDINGS', ha='center', fontsize=12, fontweight='bold')
    ax.text(4.75, 1.3, '1. Manipulation effect: [TBD after analysis]', ha='center', fontsize=9)
    ax.text(4.75, 1.0, '2. Gaming mechanism: Syntactic vs semantic', ha='center', fontsize=9)
    ax.text(4.75, 0.7, '3. Constitutional constraint prevents gaming', ha='center', fontsize=9)

    # Add data type labels
    ax.text(9, 9.5, 'DATA TYPES:', fontsize=10, fontweight='bold')
    ax.text(9, 9.2, '📄 Textual', fontsize=8)
    ax.text(9, 8.9, '🧠 Neural Repr.', fontsize=8)
    ax.text(9, 8.6, '📊 Behavioral', fontsize=8)

    plt.tight_layout()

    # Save
    output_path = 'visualizations/visual_13_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)

    print("Creating project pipeline infographic...")
    create_pipeline_visual()
    print("\n✅ Done! Created Visual 13")
