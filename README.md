# Constitutional Surprise: Can AI Game Scientific Novelty Metrics?

**Author:** Sumin Park
**Course:** AI Agents (UChicago MACSS)
**Date:** March 2026

## Overview

This project investigates whether AI agents can "game" perplexity metrics by inflating jargon without adding real substance. We apply four distinct AI techniques to 262 MACSS research abstracts:

1. **Causal Inference (DML)** - Proving manipulation causes perplexity increase
2. **Interpretability (SAE)** - Understanding how gaming works mechanistically
3. **Multi-Agent Systems** - Using debate to generate better alternatives
4. **Constitutional AI** - Preventing gaming through reward shaping

## Key Findings

- AI manipulation increases perplexity by **+72%** on average (p < 0.001)
- The effect is **causal** (proven via Double Machine Learning)
- Gaming works through **syntactic complexity**, not semantic novelty
- **Constitutional constraints** successfully prevent gaming while preserving novelty

## Repository Structure

```
constitutional-surprise/
├── data/                          # Research abstracts + perplexity scores
├── scripts/                       # Analysis pipelines
│   ├── 01_generate_manipulated.py      # GPT-3.5 jargon inflation
│   ├── 02_calculate_perplexity.py      # GPT-2 perplexity scoring
│   ├── 03_compare_perplexity.py        # Statistical comparison
│   ├── 04_causal_inference_dml.py      # Double Machine Learning
│   ├── 05_sae_analysis_simplified.py   # Sparse Autoencoder
│   ├── 06_multiagent_debate.py         # Multi-agent systems
│   └── 07_constitutional_rl.py         # Reinforcement learning
├── visualizations/                # 13 figures for blog post
├── results/                       # Model outputs (neural reps, features)
└── PRESENTATION_SLIDES.md         # 20-slide Ignite presentation

```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Running the Analysis

```bash
# 1. Generate manipulated abstracts (~30 min)
python scripts/01_generate_manipulated.py

# 2. Calculate perplexity scores (~30 min)
python scripts/03_compare_perplexity.py

# 3. Run causal inference
python scripts/04_causal_inference_dml.py

# 4. Train sparse autoencoder
python scripts/05_sae_analysis_simplified.py
```

## Methodology

### Data
- **Source:** 262 MACSS research abstracts (2015-2024)
- **Treatment:** GPT-3.5 manipulation for "maximum linguistic complexity"
- **Outcome:** Perplexity measured via GPT-2 (smaller model = more objective)

### Models
1. **DML:** Orthogonalized regression to estimate causal treatment effects
2. **SAE:** 768→256→768 autoencoder with L1 sparsity penalty
3. **Multi-agent:** 3-agent debate (Proposer, Critic, Synthesizer)
4. **Constitutional RL:** Policy gradient with gaming penalty

## Results Summary

| Metric | Original | Manipulated | Change |
|--------|----------|-------------|--------|
| Mean Perplexity | 44.1 | 76.0 | +72.3% |
| Readability (Flesch) | 42.3 | 28.7 | -32.1% |
| Semantic Content | Preserved | Preserved | ~0% |

**Interpretation:** Perplexity increased dramatically, but clarity decreased while substance remained unchanged—evidence of gaming!

## Visualizations

All 13 visualizations are available in `/visualizations/`:
- Visual 1-2: Problem demonstration
- Visual 3-5: SAE interpretability
- Visual 6-8: Causal inference results
- Visual 9-11: Constitutional RL
- Visual 12-13: Validation & pipeline

## Citation

```bibtex
@misc{park2026constitutional,
  title={Constitutional Surprise: Can AI Game Scientific Novelty Metrics?},
  author={Park, Sumin},
  year={2026},
  institution={University of Chicago}
}
```

## License

MIT License - See [LICENSE](LICENSE) file

## Acknowledgments

- **Zhen Zhang et al.** - Perplexity as novelty metric (Science, 2024)
- **Anthropic** - Constitutional AI methodology
- **Course:** AI Agents (UChicago MACSS, Winter 2026)

---

📝 **Blog Post:** \
📊 **Presentation:** See `PRESENTATION_SLIDES.md`
📧 **Contact:** [suminpark@uchicago.edu]
