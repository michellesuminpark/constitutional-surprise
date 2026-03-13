# Constitutional Surprise: Can AI Game the Perplexity Metric?
## 20-Slide Presentation (5 minutes / 15 sec per slide)

---

## SLIDE 1: Title
**Constitutional Surprise: Can AI Agents Game Scientific Novelty Metrics?**

Sumin Park
AI Agents Class - Final Project
March 13, 2026

---

## SLIDE 2: The Perplexity Paradox
**Visual:** visual_01_text_comparison.png

**Key Point:** Higher perplexity = more "surprising" text... but is it better quality?

---

## SLIDE 3: Research Question
**Can AI agents make research "more surprising" without making it better?**

- UChicago motto: "Crescat Scientia" (let knowledge grow)
- Problem: Most research confirms the expected (boring!)
- Solution: Use AI to increase novelty?
- Risk: What if AI just "games" the metric?

---

## SLIDE 4: Data Collection
**270 MACSS Abstracts**

Three data types:
1. 📄 **Textual:** Original + manipulated abstracts
2. 🧠 **Neural:** GPT-2 activations (768-dim)
3. 📊 **Behavioral:** Perplexity scores

---

## SLIDE 5: Methods Overview
**Visual:** visual_13_pipeline.png

**4 Models from 4 Different Weeks:**
1. GPT-3.5 (Week 3: Agents)
2. GPT-2 (Week 5: Language Models)
3. DML (Week 4: Causal Inference)
4. SAE (Week 6: Interpretability)

---

## SLIDE 6: Model 1 - Text Manipulation
**Agent-based manipulation (GPT-3.5)**

Prompt: "Maximize linguistic complexity"
- Use rare vocabulary
- Create unexpected word combinations
- Add cross-disciplinary jargon
- Temperature = 0.9 for variation

---

## SLIDE 7: Perplexity Measurement
**GPT-2 calculates "surprise"**

Formula: `Perplexity = exp(negative_avg_log_prob)`

- Low perplexity = predictable text
- High perplexity = surprising text
- Used as proxy for "novelty"

---

## SLIDE 8: The Results Are In!
**Visual:** visual_02_correlation_scatter.png

**DRAMATIC EFFECT:**
- Original: 44.1 mean perplexity
- Manipulated: 76.0 mean perplexity
- **+72.3% increase!** (p < 0.001)

---

## SLIDE 9: Model 2 - Causal Inference
**Visual:** visual_06_causal_dag.png

**Question:** Does manipulation CAUSE perplexity increase?

**Method:** Double Machine Learning (DML)
- Controls for confounders (length, field, baseline)
- Orthogonalized regression

---

## SLIDE 10: DML Results - Causality Proven!
**Visual:** visual_07_treatment_effect_forest.png

**Average Treatment Effect:**
- ATE = **+54.62 points**
- 95% CI: [50.81, 58.44]
- t-statistic: 28.07
- **p < 0.001** ✓✓✓

Manipulation CAUSES perplexity increase!

---

## SLIDE 11: Heterogeneous Effects
**Visual:** visual_08_cate_heterogeneity.png

**Not all papers respond equally:**
- Low baseline: +40 points effect
- High baseline: +17.5 points effect
- **2.28x more effective on "boring" papers!**

---

## SLIDE 12: Model 3 - Interpretability (SAE)
**Visual:** visual_03_sae_architecture.png

**Question:** HOW does manipulation work?

**Sparse Autoencoder:**
- Input: 768-dim GPT-2 activations
- Hidden: 256-dim sparse features
- Sparsity constraint: L1 penalty

---

## SLIDE 13: SAE Features Discovered
**Visual:** visual_04_feature_heatmap.png

**Top 10 features correlate with perplexity change**

Key insight: Can separate:
- Syntactic complexity (surface-level)
- Semantic novelty (real content)

---

## SLIDE 14: Feature Space Visualization
**Visual:** visual_05_feature_pca.png

**3D projection shows:**
- Original abstracts cluster together (green)
- Manipulated abstracts more scattered (red)
- High variance in manipulation effects

---

## SLIDE 15: The Gaming Problem
**Is this just syntactic gaming?**

Evidence:
- Manipulation adds jargon
- Adds complex nested clauses
- Uses rare vocabulary
- BUT: Does it add real insight?

**Answer:** Perplexity increased, but...

---

## SLIDE 16: Model 4 - Constitutional AI
**Visual:** visual_09_reward_function.png

**Solution:** Add penalty for gaming

- Baseline: R = -Perplexity
- Constitutional: R = -Perplexity + λ·F_gaming

Penalty prevents syntactic gaming!

---

## SLIDE 17: Multi-Agent Alternative
**Visual:** visual_10_multiagent_debate.png

**Three agents collaborate:**
1. Proposer: "Add jargon"
2. Critic: "Too much gaming!"
3. Synthesizer: "Balance novelty & clarity"

Result: Better abstracts through debate

---

## SLIDE 18: RL Training Shows Constraint Works
**Visual:** visual_11_rl_trajectories.png

**Simulated training:**
- Baseline agent: ↑ syntactic features
- Constitutional agent: ↑ semantic features

Constraint successfully redirects learning!

---

## SLIDE 19: Quality vs Novelty Trade-off
**Visual:** visual_12_quality_metrics.png

**Key Finding:**
- Novelty ↑ (perplexity increased)
- Clarity ↓ (harder to read)
- Substance ≈ (preserved)

**Implication:** Gaming is real!

---

## SLIDE 20: Conclusions & Future Work
**What We Learned:**

✅ AI CAN game perplexity metrics (+72.3%)
✅ Effect is causal (DML proves it)
✅ Gaming is primarily syntactic (SAE shows how)
✅ Constitutional constraints can prevent gaming

**Limitation:** Doesn't address scaffolding vs direct answers

**Future:** Test whether this helps students think critically

---

## THANK YOU!
Questions?

Contact: your-email@uchicago.edu
Code: github.com/your-username/constitutional-surprise

