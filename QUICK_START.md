# 🚀 QUICK START - Constitutional Surprise

**Current time: March 12, ~6pm**
**Deadline: March 13, 3pm (slides) / 5pm (blog)**
**Time remaining: ~21 hours**

---

## ⚡ START HERE - First 30 Minutes

### Step 1: Verify Python environment
```bash
cd /Users/suminpark/Documents/GitHub/uchicago-wi26/ai_agents-class/constitutional-surprise

# Check if packages are installed
python -c "import pandas, torch, transformers, openai; print('✅ Core packages OK')"
```

**If missing packages:**
```bash
pip install pandas torch transformers openai tqdm econml sklearn networkx matplotlib seaborn plotly autogen
```

### Step 2: Set OpenAI API key
```bash
export OPENAI_API_KEY="your-key-here"
# Or add to .env file
```

### Step 3: Run data generation pipeline
```bash
# This will take ~1 hour total
python scripts/01_generate_manipulated.py
# Wait for completion, then:
python scripts/03_compare_perplexity.py
```

**While scripts are running, start working on:**
- Causal inference code (script 04)
- SAE analysis code (script 05)

---

## 📋 Hour-by-Hour Battle Plan

### HOUR 1-2 (6pm-8pm): DATA GENERATION
**Do now:**
- ✅ Run scripts 01 and 03
- Start drafting DML code
- Create project structure folders

### HOUR 3-5 (8pm-11pm): MODEL 1 & 2
**Do now:**
- Implement DML (causal inference)
- Implement SAE analysis
- Generate first 6 visualizations

### HOUR 6-8 (11pm-1am): MODEL 3 & 4
**Do now:**
- Multi-agent debate implementation
- Start RL implementation
- Generate visualizations 7-11

### HOUR 9-10 (1am-3am): VALIDATION
**Do now:**
- Human validation setup (or LLM proxy)
- Generate visualization 12
- Create pipeline infographic (visual 13)

**SLEEP** (3am-7am) - 4 hours

### HOUR 11-14 (7am-10am): BLOG POST
**Do now:**
- Write 3600 words
- Embed all 13 visualizations
- Proofread

### HOUR 15-17 (10am-1pm): PRESENTATION
**Do now:**
- Create 20 slides
- Practice timing (15 sec/slide)
- Export to PDF

### HOUR 18-21 (1pm-4pm): POLISH & SUBMIT
**Do now:**
- Final review
- Submit slides by 3pm ✅
- Submit blog post by 5pm ✅

---

## 🎯 Minimum Viable Product (If Time Runs Out)

### Must Have (80% effort → 90% credit):
1. ✅ Complete dataset with perplexity scores
2. ✅ DML causal analysis + 3 visuals (6,7,8)
3. ✅ SAE interpretability + 3 visuals (3,4,5)
4. ✅ Multi-agent (simple debate between GPT-3.5 and GPT-4)
5. ✅ Blog post: 3600 words + 13 visuals
6. ✅ Presentation: 20 slides

### Nice to Have (20% effort → 10% credit):
- Full RL implementation
- Human validation (vs LLM proxy)
- Polished graphics
- Extended analysis

---

## 🆘 Emergency Contacts & Resources

### If stuck on DML:
- Week 4 tutorial: `/Tutorials-Homework_Notebooks/Week_4/week_4_2026.ipynb`
- Existing script: `week4_real_analysis.py`

### If stuck on SAE:
- Week 6 tutorial: `/Tutorials-Homework_Notebooks/Week_6/week6_2026.ipynb`
- Related: `Perplexity and Linguistic Novelty.ipynb`

### If stuck on multi-agent:
- Week 3 tutorial: `/Tutorials-Homework_Notebooks/Week_3/Week_3.ipynb`
- Existing: `surprise_interventions_colab.py`

### If stuck on RL:
- Week 7 tutorial: `/Tutorials-Homework_Notebooks/Week_7/Week_7_RL.ipynb`
- Related: `Serendipity_Agent_Professional.ipynb`

---

## 📊 Progress Tracking

Mark as you complete:

**Data (Phase 1):**
- [ ] Manipulated abstracts generated
- [ ] Perplexity calculated for all
- [ ] Dataset validated (no missing values)

**Models (Phase 2):**
- [ ] DML implementation complete
- [ ] SAE implementation complete
- [ ] Multi-agent implementation complete
- [ ] RL implementation complete (or skipped)

**Visuals (Phase 4):**
- [ ] Visual 1: Provocation
- [ ] Visual 2: Correlation scatter
- [ ] Visual 3: SAE architecture
- [ ] Visual 4: Feature heatmap
- [ ] Visual 5: 3D gaming scatter
- [ ] Visual 6: Causal DAG
- [ ] Visual 7: Treatment effect forest plot
- [ ] Visual 8: CATE plot
- [ ] Visual 9: Reward function
- [ ] Visual 10: Debate flow
- [ ] Visual 11: RL trajectories
- [ ] Visual 12: Human ratings
- [ ] Visual 13: Pipeline infographic

**Deliverables (Phases 5-6):**
- [ ] Blog post drafted (3600+ words)
- [ ] Blog post proofread
- [ ] Presentation slides created (20 slides)
- [ ] Presentation timed (5 min)
- [ ] Slides submitted (by 3pm)
- [ ] Blog post submitted (by 5pm)

---

## 💻 Quick Code Templates

### Template: Load perplexity data
```python
import pandas as pd

df = pd.read_csv("data/macss_full_with_perplexity.csv")
print(f"Loaded {len(df)} abstracts")
print(f"Columns: {df.columns.tolist()}")
```

### Template: Calculate treatment effect
```python
treatment_effect = df['perplexity_manipulated'] - df['perplexity_original']
print(f"Mean effect: {treatment_effect.mean():.2f}")
print(f"Effect size: {treatment_effect.mean() / df['perplexity_original'].mean() * 100:.1f}%")
```

### Template: Create visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='perplexity_original', y='perplexity_manipulated')
plt.xlabel('Original Perplexity')
plt.ylabel('Manipulated Perplexity')
plt.title('Treatment Effect: Gaming Increases Perplexity')
plt.savefig('visualizations/visual_02_correlation.png', dpi=300, bbox_inches='tight')
```

---

## 🎓 Blog Post Template (Copy & Fill)

```markdown
# Can Boring Research Be Made More Surprising? A Constitutional AI Experiment

## Part I: The Problem (600 words)

What if AI makes research more "surprising" without making it better?

[Visual 1 here]

[Your text here...]

## Part II: Dissecting the Mechanism (900 words)

Tutorial: How to reverse-engineer surprise with sparse autoencoders

[Visual 3 here]
[Visual 4 here]
[Visual 5 here]

[Your text here...]

## Part III: Proving Causality (700 words)

Did the agent really cause the change?

[Visual 6 here]
[Visual 7 here]
[Visual 8 here]

[Your text here...]

## Part IV: Generating Real Surprise (900 words)

Building a constitutional AI that can't cheat

[Visual 9 here]
[Visual 10 here]
[Visual 11 here]

[Your text here...]

## Part V: Validation (600 words)

Does this actually work?

[Visual 12 here]
[Visual 13 here]

[Your text here...]

## Part VI: Implications (900 words)

What this means for AI-assisted research

[Your text here with callout boxes...]
```

---

**YOU GOT THIS! 🚀**

Focus, execute, ship. Quality over perfection.
The roadmap is your guide. Follow it sequentially.

When in doubt: **DML first, SAE second, blog post third.**
