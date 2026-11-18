# Fairness-Aware Continual Learning for E-commerce Recommendation Systems
**Phase 1 Prototype & Interactive Streamlit Dashboard**

---

> ⚠️ **DISCLAIMER & DEMONSTRATION NOTICE**:  
> This Phase 1 prototype operates on a **controlled, reproducible synthetic e-commerce interaction dataset** generated locally. It is designed to illustrate continual learning adaptation, feedback loops, fairness drift detection, and post-processing reranking without requiring heavy external Amazon Review downloads. **All data and results are simulated and clearly marked as synthetic demonstration data.**

---

## 🎯 Project Overview & Review Objectives

Online recommendation systems in e-commerce continuously update on incoming streams of user interactions. While continuous retraining preserves recommendation accuracy, it frequently suffers from **catastrophic forgetting of fairness** — where algorithmic feedback loops and popularity bias cause recommendation quality for underrepresented customer groups to silently degrade over successive retraining rounds.

This working Phase 1 prototype provides:
1. **Interactive E-Commerce Recommendation Interface**: Product recommendations with user selector, category badges, predicted relevance scores, and side-by-side comparison between **Base Model** and **Fairness-Aware Reranked Model**.
2. **Stateful Continual Learning Simulation**: Ingests interaction batches chronologically using online Stochastic Gradient Descent Matrix Factorization (`river.reco.BiasedMF`).
3. **Recommendation Quality Evaluation**: Live calculations of **Precision@K**, **Recall@K**, and **NDCG@K**.
4. **Group Fairness Monitoring**: Tracks **Demographic Parity Difference**, **Disparate Impact Ratio (80% Rule)**, and **Equal Opportunity Difference (Recall Gap)** between two documented demonstration groups:
   - **Group A (50%)**: Standard / Emerging Shoppers.
   - **Group B (50%)**: Premium / Core Shoppers.
5. **Configurable Fairness Threshold & Reranking Mitigation**: Alerts when fairness drift exceeds tolerance $\epsilon$, activating a post-processing reranking algorithm that restores group parity.
6. **Chronological Analytics Charts**: Dynamic visualization showing quality and fairness progression across all completed rounds.

---

## 📂 Project Structure

```text
clfbrs/
│
├── app.py                      # Main Streamlit dashboard application
├── run_dashboard.bat           # 1-click Windows launcher
├── README.md                   # Complete documentation & viva talking points
│
├── src/                        # Modular system components
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic e-commerce dataset (100 users, 40 items, 6 batches)
│   ├── recommender.py          # Continual recommender (River online BiasedMF)
│   ├── fairness.py             # Fairness metrics (DP, DI, EO) & Fairness-Preserving Reranker
│   ├── evaluation.py           # Precision@K, Recall@K, NDCG@K, HitRate@K
│   └── simulation.py           # Batch controller, stateful session coordinator
│
└── .venv/                      # Python 3.12 Virtual Environment
```

---

## 🚀 How to Run the Prototype Locally

### Option 1: One-Click Launcher (Windows)
Double-click `run_dashboard.bat` in the project root folder.

### Option 2: Command Line (PowerShell / CMD)
```powershell
# Navigate to the project directory
cd c:\Users\Dell\Desktop\clfbrs

# Launch the Streamlit application using the existing .venv
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 📊 Mathematical Formulation

### 1. Continual Learning Update Step
For each interaction record $(u, i, r)$ in chronological batch $B_t$:
$$\theta_t = \theta_{t-1} - \eta \cdot \nabla \mathcal{L}(B_t; \theta_{t-1})$$
Model parameters update observation-by-observation without retraining from scratch.

### 2. Fairness Metrics
- **Demographic Parity Difference ($\Delta_{\text{DP}}$)**:
  $$\Delta_{\text{DP}} = |\text{Precision@K}(G=B) - \text{Precision@K}(G=A)|$$
- **Disparate Impact Ratio ($\text{DI}$)**:
  $$\text{DI} = \frac{\min(\text{Precision@K}(G=A), \text{Precision@K}(G=B))}{\max(\text{Precision@K}(G=A), \text{Precision@K}(G=B))}$$
  *Acceptance criteria:* $\text{DI} \ge 0.80$ (The four-fifths rule).
- **Equal Opportunity Difference ($\Delta_{\text{EO}}$)**:
  $$\Delta_{\text{EO}} = |\text{Recall@K}(G=B) - \text{Recall@K}(G=A)|$$

### 3. Fairness-Preserving Reranking Mechanism
When $\Delta_{\text{DP}} > \epsilon$ or $\text{DI} < 0.80$, candidate items are reranked via score blending:
$$S_{\text{reranked}}(u, i) = (1 - \alpha) \cdot S_{\text{pred}}(u, i) + \alpha \cdot S_{\text{fairness\_boost}}(u, i, G_u)$$
where $\alpha \in [0, 1]$ represents the mitigation intensity configured in the sidebar.

---

## 🔬 System Capabilities

- **Incremental Adaptation**: Updates continuously on user interaction batches using online Matrix Factorization without retraining from scratch.
- **Fairness Drift Monitoring**: Continuously monitors Demographic Parity and Disparate Impact across customer segments.
- **Dynamic Reranking**: Applies real-time post-processing reranking when fairness thresholds are violated.
- **Performance Audit**: Tracks Precision@K, Recall@K, and NDCG@K across all retraining rounds.
