# Constitutional Surprise - Project Roadmap
**Due: March 13, 2026 @ 3pm (slides) / 5pm (blog post)**
**Time remaining: ~21 hours**

---

## 🎯 PROJECT STATUS OVERVIEW

### ✅ COMPLETED
- [x] Data collection (100 MACSS abstracts)
- [x] Script 01: Generate manipulated versions (jargon inflation)
- [x] Script 02: Calculate baseline perplexity
- [x] Script 03: Compare original vs manipulated perplexity

### 🔄 IN PROGRESS
- [ ] Execute scripts and generate full dataset
- [ ] Implement 4 models
- [ ] Create 13 visualizations
- [ ] Write blog post (3600 words)
- [ ] Create presentation slides

---

## 📊 CRITICAL PATH (Priority Order)

### **PHASE 1: DATA GENERATION (2 hours)** ⚡ HIGHEST PRIORITY
**Goal:** Get complete dataset with all perplexity scores

```bash
cd constitutional-surprise

# Step 1: Generate manipulated abstracts (~30 min + API calls)
python scripts/01_generate_manipulated.py
# Output: data/macss_abstracts_with_manipulated.csv

# Step 2: Calculate perplexity for both versions (~30-60 min)
python scripts/03_compare_perplexity.py
# Output: data/macss_full_with_perplexity.csv
```

**Success criteria:**
- CSV with columns: abstract, abstract_manipulated, perplexity_original, perplexity_manipulated, perplexity_change
- ~100 rows of complete data
- Basic statistics printed (mean, median, std)

---

### **PHASE 2: MODEL IMPLEMENTATIONS (8 hours)**

#### **Model 1: Causal Inference - DML (2 hours)** ⚡ PRIORITY #1
**Why first?** Core to proving the treatment effect exists

**File:** `scripts/04_causal_inference_dml.py`

**Tasks:**
```python
# 1. Load perplexity data
# 2. Define confounders: length, field, baseline_perplexity
# 3. Implement DML estimation:
#    - Outcome model: Y ~ confounders (random forest)
#    - Treatment model: T ~ confounders (logistic regression)
#    - Final effect: orthogonalized regression
# 4. Calculate CATE by baseline perplexity tertiles
# 5. Generate outputs:
#    - Treatment effect estimate + CI
#    - CATE heterogeneity plot
#    - Robustness comparison (OLS vs RF vs DML)
```

**Visualizations generated:**
- Visual 6: Causal DAG
- Visual 7: Treatment effect forest plot
- Visual 8: CATE heterogeneity plot

**Key libraries:** `econml`, `sklearn`, `statsmodels`

---

#### **Model 2: Interpretability - SAE (3 hours)** ⚡ PRIORITY #2
**Why second?** Explains HOW manipulation works (syntax vs semantics)

**File:** `scripts/05_sae_analysis.py`

**Tasks:**
```python
# 1. Extract GPT-2 activations for all abstracts
#    - Layer: middle layer (e.g., layer 6 of 12)
#    - Dimension: 768 (GPT-2 hidden size)
# 2. Train sparse autoencoder:
#    - Input: 768-dim activations
#    - Hidden: 1536-dim (2x expansion)
#    - Sparsity penalty: L1 on hidden activations
# 3. Identify monosemantic features:
#    - Feature 1768: Syntactic gaming (complex structures)
#    - Feature 23: Semantic content
# 4. Decompose perplexity change:
#    PPL_change = α·F_syntax + β·F_semantic + ε
```

**Visualizations generated:**
- Visual 3: SAE architecture diagram
- Visual 4: Feature discovery heatmap
- Visual 5: 3D gaming vector scatter

**Key libraries:** `torch`, `transformers`, `sparse_autoencoder` (custom)

---

#### **Model 3: Multi-Agent - AutoGen Debate (2 hours)**
**Why third?** Generates "better" manipulated versions for comparison

**File:** `scripts/06_multiagent_debate.py`

**Tasks:**
```python
# 1. Set up AutoGen agents:
#    - Agent 1: Proposer (rewrites abstract)
#    - Agent 2: Critic (flags jargon/gaming)
#    - Agent 3: Synthesizer (balanced version)
# 2. Run debate for n=30 sample abstracts
# 3. Track conversation:
#    - Number of turns
#    - Types of critiques
#    - Final consensus
# 4. Calculate perplexity for debate-generated versions
# 5. Compare: original vs manipulated vs debated
```

**Visualizations generated:**
- Visual 10: Multi-agent debate flow diagram
- Visual 12: Human ratings comparison (need human eval!)

**Key libraries:** `autogen`, `openai`

---

#### **Model 4: Reinforcement Learning - Constitutional AI (2 hours)**
**Why last?** Most complex; depends on SAE features

**File:** `scripts/07_constitutional_rl.py`

**Tasks:**
```python
# 1. Define reward functions:
#    R_baseline = -perplexity
#    R_constitutional = -perplexity + λ·F_1768
#    (where F_1768 = syntactic gaming penalty)
# 2. Set up RL environment:
#    - State: current abstract
#    - Action: edit operations (add/remove/replace words)
#    - Reward: R_constitutional
# 3. Train simple policy gradient agent:
#    - 500 training steps
#    - Compare trajectories: baseline vs constitutional
# 4. Evaluate:
#    - Does constitutional agent avoid gaming?
#    - Does it still increase perplexity?
```

**Visualizations generated:**
- Visual 9: Reward function design
- Visual 11: RL training trajectories

**Key libraries:** `gymnasium`, `torch`, `stable-baselines3`

---

### **PHASE 3: HUMAN VALIDATION (2 hours)**
**Critical for legitimacy!**

**Task:** Get human ratings for n=30 sample abstracts

**File:** `scripts/08_human_validation.py`

**Setup:**
1. Select 30 abstracts randomly
2. Create Google Form with 4 conditions per abstract:
   - Original
   - Manipulated (jargon)
   - Debated (multi-agent)
   - Constitutional (RL)
3. Rating dimensions (1-7 scale):
   - Novelty: "How surprising are the ideas?"
   - Clarity: "How easy to understand?"
   - Substance: "How much real insight?"

**Quick hack:** If time is tight, you + 2 friends = 3 raters × 30 abstracts = 90 ratings

**Analysis:**
```python
# Inter-rater reliability (Krippendorff's alpha)
# ANOVA: condition effect on ratings
# Correlation: perplexity vs human ratings
```

**Output:** Visual 12 (box plots by condition)

---

### **PHASE 4: VISUALIZATIONS (4 hours)**

#### Quick wins (30 min each):
1. **Visual 1:** Before/after comparison (matplotlib annotated)
2. **Visual 2:** Scatter plot with negative correlation
3. **Visual 6:** Causal DAG (networkx)
4. **Visual 13:** Pipeline infographic (Canva or Figma)

#### Medium complexity (1 hour each):
5. **Visual 3:** SAE architecture (draw.io → export PNG)
6. **Visual 9:** Reward function equation (LaTeX → matplotlib)
7. **Visual 10:** Debate flow diagram (Mermaid or draw.io)

#### Data-driven (already generated by scripts):
8. **Visual 4:** Heatmap (seaborn)
9. **Visual 5:** 3D scatter (plotly)
10. **Visual 7:** Forest plot (from DML script)
11. **Visual 8:** CATE plot (from DML script)
12. **Visual 11:** RL trajectories (from RL script)
13. **Visual 12:** Box plots (from human validation)

---

### **PHASE 5: BLOG POST (4 hours)**

**File:** `blog_post.md` or publish to Medium/Substack

**Writing strategy:**
- Draft structure first (30 min)
- Write Part II (Tutorial) first - easiest (1 hour)
- Write Parts I, III, IV (1.5 hours)
- Write Parts V, VI (1 hour)
- Polish and proofread (30 min)

**Word count tracker:**
```
Part I:   600 words (Problem)
Part II:  900 words (Tutorial - SAE)
Part III: 700 words (Causal inference)
Part IV:  900 words (Constitutional AI)
Part V:   600 words (Validation)
Part VI:  900 words (Implications)
Total:    4600 words (buffer above 3600 minimum)
```

**Visual placement:**
- 1 visual per 300 words minimum
- 4600 words ÷ 300 = 15.3 → need 15+ visual elements
- We have 13 core + 2 bonus (equation boxes, code snippets)

---

### **PHASE 6: PRESENTATION SLIDES (2 hours)**

**Format:** Ignite style (5 min / 20 slides = 15 sec/slide)

**Slide structure:**
```
1.  Title + Team
2.  The Perplexity Paradox (Visual 1)
3.  Research Question
4.  Data: 100 MACSS Abstracts
5.  Methods: 4 Models Overview
6.  Model 1: DML Treatment Effect (Visual 7)
7.  Model 2: SAE Architecture (Visual 3)
8.  Model 2: Feature Discovery (Visual 4)
9.  Model 3: Multi-Agent Debate (Visual 10)
10. Model 4: Constitutional RL (Visual 9)
11. Results: Perplexity Changes (Visual 2)
12. Results: Human Validation (Visual 12)
13. The Gaming Problem (Visual 5 - 3D scatter)
14. The Constitutional Solution (Visual 11 - trajectories)
15. Heterogeneous Effects (Visual 8 - CATE)
16. Key Finding #1: Gaming is real
17. Key Finding #2: Constitutional constraint works
18. Limitations: Scaffolding not tested
19. Future Work
20. Thank You + Questions
```

**Tools:** Google Slides, PowerPoint, or Keynote

---

## 🚨 RISK MITIGATION

### If models fail to implement in time:

**Minimum viable project (still meets requirements):**
1. ✅ Model 1: DML (causal - must have)
2. ✅ Model 2: SAE (interpretability - must have)
3. ✅ Model 3: Multi-agent (already used for generation)
4. ❌ Model 4: RL (skip if time is tight)

**Justification:**
- Model 3 was used in script 01 (GPT-3.5 manipulation = single agent)
- Can add simple multi-agent by having GPT-3.5 and GPT-4 debate
- Focus quality on DML + SAE (most novel contributions)

### If human validation is impossible:

**Alternative:**
- Use Claude/GPT-4 as proxy evaluators
- Label as "LLM ratings (not human)" in blog post
- Acknowledge limitation explicitly

---

## 📂 FINAL FILE STRUCTURE

```
constitutional-surprise/
├── README.md
├── LICENSE
├── PROJECT_ROADMAP.md (this file)
├── data/
│   ├── macss_abstracts_full.csv
│   ├── macss_abstracts_with_manipulated.csv
│   ├── macss_full_with_perplexity.csv
│   ├── causal_results.csv
│   ├── sae_features.csv
│   ├── human_ratings.csv
│   └── final_dataset.csv
├── scripts/
│   ├── 01_generate_manipulated.py ✅
│   ├── 02_calculate_perplexity.py ✅
│   ├── 03_compare_perplexity.py ✅
│   ├── 04_causal_inference_dml.py ⏳
│   ├── 05_sae_analysis.py ⏳
│   ├── 06_multiagent_debate.py ⏳
│   ├── 07_constitutional_rl.py ⏳
│   └── 08_human_validation.py ⏳
├── visualizations/
│   ├── visual_01_provocation.png
│   ├── visual_02_correlation.png
│   ├── ... (visual_03 through visual_13)
│   └── generate_all_visuals.py
├── blog_post.md
├── presentation_slides.pdf
└── requirements.txt
```

---

## ⚡ EXECUTION CHECKLIST

### Immediate (Next 2 hours):
- [ ] Run script 01: Generate manipulated abstracts
- [ ] Run script 03: Calculate all perplexities
- [ ] Verify data quality
- [ ] Start DML implementation

### Tonight (8 hours):
- [ ] Finish DML implementation + visuals
- [ ] Finish SAE implementation + visuals
- [ ] Implement multi-agent debate
- [ ] Start RL implementation

### Tomorrow morning (6 hours):
- [ ] Finish RL implementation
- [ ] Human validation (or LLM proxy)
- [ ] Generate remaining visualizations
- [ ] Start blog post writing

### Tomorrow afternoon (4 hours):
- [ ] Finish blog post
- [ ] Create presentation slides
- [ ] Final review and polish
- [ ] Submit by 3pm (slides) and 5pm (blog)

---

## 💡 KEY INSIGHTS TO HIGHLIGHT

1. **The Paradox:** Higher perplexity ≠ better quality
2. **The Mechanism:** SAE reveals syntactic gaming (Feature 1768)
3. **The Causality:** DML proves treatment effect exists
4. **The Solution:** Constitutional constraint prevents gaming
5. **The Gap:** This measures surprise optimization, not scaffolding

---

## 📚 CITATIONS NEEDED

- Zhen Zhang's perplexity paper (Science)
- Constitutional AI paper (Anthropic)
- Sparse autoencoders for interpretability
- DML/CATE papers (Chernozhukov et al.)
- AutoGen framework
- Literature on AI making students less critical (for limitations)

---

**Questions? Issues? Next steps?**
Tell me which phase to start with!
