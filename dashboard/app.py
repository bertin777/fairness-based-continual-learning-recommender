"""
Fairness-Aware Continual Learning: Building Recommendation Systems for E-commerce Websites
That Adapt Without Losing Fairness.

Phase 1 Working Prototype - Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import time

from src.data_generator import generate_synthetic_dataset, CATEGORY_ICONS, PRODUCTS_CATALOG
from src.simulation import ContinualSimulationManager
from src.evaluation import evaluate_recommendations_at_k
from src.fairness import FairnessReranker, FairnessEvaluator


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fairness-Aware Continual Recommender",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .badge-demo {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FCD34D;
        margin-bottom: 0.8rem;
    }
    .card-metric {
        background: #F8FAFC;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-metric h4 {
        margin: 0 0 6px 0;
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card-metric .val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .card-metric .sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }
    .product-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        transition: transform 0.15s ease;
    }
    .product-card:hover {
        border-color: #94A3B8;
        background: #F8FAFC;
    }
    .badge-cat {
        background: #EEF2FF;
        color: #3730A3;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .badge-group-a {
        background: #E0E7FF;
        color: #3730A3;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .badge-group-b {
        background: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    .alert-box-violated {
        background: #FEE2E2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        margin: 10px 0;
    }
    .alert-box-ok {
        background: #DCFCE7;
        border-left: 5px solid #22C55E;
        color: #166534;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------
if "simulation" not in st.session_state:
    st.session_state.simulation = ContinualSimulationManager(
        n_users=100,
        n_batches=6,
        k=5,
        fairness_threshold=0.10,
        min_disparate_impact=0.80,
        reranking_alpha=0.35,
        seed=42
    )
    # Initialize round 1 automatically so the dashboard starts with initial state
    st.session_state.simulation.process_next_batch()

sim: ContinualSimulationManager = st.session_state.simulation


# -----------------------------------------------------------------------------
# Sidebar: Controls & Parameter Tuning
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation Configuration")
    
    k_val = st.slider(
        "Top-K Recommendations (Cutoff)",
        min_value=3,
        max_value=10,
        value=sim.k,
        step=1,
        help="Number of recommendations evaluated per customer"
    )
    
    threshold_val = st.slider(
        "Fairness Tolerance Threshold (ε)",
        min_value=0.04,
        max_value=0.25,
        value=sim.fairness_threshold,
        step=0.01,
        help="Maximum allowable Demographic Parity Difference |Group B - Group A| before triggering alert"
    )

    min_di_val = st.slider(
        "Minimum Disparate Impact Ratio",
        min_value=0.65,
        max_value=0.95,
        value=sim.min_disparate_impact,
        step=0.05,
        help="Legal/Academic 80% Rule threshold: min(SR_A, SR_B) / max(SR_A, SR_B)"
    )

    alpha_val = st.slider(
        "Reranking Fairness Weight (α)",
        min_value=0.0,
        max_value=0.80,
        value=sim.reranking_alpha,
        step=0.05,
        help="Weight allocated to fairness correction component during reranking (0.0 = Raw Model, >0.3 = Fair Reranked)"
    )

    # Sync parameters with simulation manager
    sim.set_parameters(
        k=k_val,
        fairness_threshold=threshold_val,
        min_disparate_impact=min_di_val,
        reranking_alpha=alpha_val
    )

    st.markdown("---")
    st.markdown("### 🔄 Continual Learning Batches")
    
    rounds_done = sim.current_batch_idx
    total_rounds = len(sim.batches)
    st.write(f"**Current Progress:** Round {rounds_done} of {total_rounds}")
    st.progress(rounds_done / total_rounds)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⏭️ Next Batch", use_container_width=True, disabled=not sim.has_next_batch()):
            with st.spinner("Processing batch incrementally..."):
                sim.process_next_batch()
                st.rerun()

    with col_btn2:
        if st.button("⚡ Fast Forward", use_container_width=True, disabled=not sim.has_next_batch()):
            with st.spinner("Processing all batches..."):
                sim.run_all_batches()
                st.rerun()

    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.simulation = ContinualSimulationManager(
            n_users=100,
            n_batches=6,
            k=k_val,
            fairness_threshold=threshold_val,
            min_disparate_impact=min_di_val,
            reranking_alpha=alpha_val,
            seed=42
        )
        st.session_state.simulation.process_next_batch()
        st.rerun()

    st.markdown("---")
    st.markdown("### 👥 Demonstration Groups")
    st.markdown("""
    - <span class="badge-group-a">Group A (50%)</span>: Standard / Emerging Shoppers.
    - <span class="badge-group-b">Group B (50%)</span>: Premium / Core Shoppers.
    
    **Continual Bias Mechanism:**
    Feedback loops in rounds 3-6 skew interaction volume towards Group B, causing standard models to suffer fairness drift against Group A.
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Header & Overview
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">⚖️ Fairness-Aware Continual Learning Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Building Recommendation Systems for E-commerce Websites That Adapt Incrementally Without Losing Fairness</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Current Round Summary & Drift Status Banner
# -----------------------------------------------------------------------------
latest_snapshot = sim.round_history[-1] if sim.round_history else None
if latest_snapshot:
    raw_f = latest_snapshot["raw_fairness"]
    fair_f = latest_snapshot["fair_fairness"]
    raw_q = latest_snapshot["raw_quality"]["overall"]
    fair_q = latest_snapshot["fair_quality"]["overall"]

    # Status Banner
    if raw_f["is_violated"]:
        st.markdown(f"""
        <div class="alert-box-violated">
            🚨 <b>FAIRNESS DRIFT ALERT DETECTED IN ROUND {latest_snapshot['round']}!</b><br>
            The unmitigated continual model has drifted beyond your configured tolerance (Demographic Parity Diff = {raw_f['demographic_parity_diff']:.3f} > ε = {raw_f['threshold_configured']:.2f}, Disparate Impact = {raw_f['disparate_impact_ratio']:.3f} < 0.80).<br>
            ✨ <i>Fairness-Preserving Reranking is active to restore group parity!</i>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box-ok">
            ✅ <b>FAIRNESS COMPLIANT (ROUND {latest_snapshot['round']}):</b> 
            Demographic Parity Difference ({raw_f['demographic_parity_diff']:.3f}) is within tolerance threshold (ε = {raw_f['threshold_configured']:.2f}). Disparate Impact is {raw_f['disparate_impact_ratio']:.3f}.
        </div>
        """, unsafe_allow_html=True)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="card-metric">
            <h4>Learning Round</h4>
            <div class="val">Round {latest_snapshot['round']} <span style="font-size: 1rem; color: #64748B;">/ {len(sim.batches)}</span></div>
            <div class="sub">Total Interactions: <b>{latest_snapshot['total_interactions']}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card-metric">
            <h4>Recommendation Quality</h4>
            <div class="val">{raw_q['precision_at_k']:.3f} <span style="font-size: 0.9rem; color: #64748B;">P@{sim.k}</span></div>
            <div class="sub">Recall@{sim.k}: <b>{raw_q['recall_at_k']:.3f}</b> | NDCG@{sim.k}: <b>{raw_q['ndcg_at_k']:.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        drift_color = "#DC2626" if raw_f["is_violated"] else "#16A34A"
        st.markdown(f"""
        <div class="card-metric">
            <h4>Demographic Parity Diff</h4>
            <div class="val" style="color: {drift_color};">{raw_f['demographic_parity_diff']:.3f}</div>
            <div class="sub">Threshold ε: <b>{raw_f['threshold_configured']:.2f}</b> | Fair Model: <b>{fair_f['demographic_parity_diff']:.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        di_color = "#DC2626" if raw_f["disparate_impact_ratio"] < raw_f["min_disparate_impact_threshold"] else "#16A34A"
        st.markdown(f"""
        <div class="card-metric">
            <h4>Disparate Impact Ratio</h4>
            <div class="val" style="color: {di_color};">{raw_f['disparate_impact_ratio']:.3f}</div>
            <div class="sub">80% Rule Target: <b>≥ 0.80</b> | Fair Model: <b>{fair_f['disparate_impact_ratio']:.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_recs, tab_analytics, tab_mitigation, tab_audit = st.tabs([
    "🛍️ E-Commerce Recommender",
    "📈 Continual Learning Analytics",
    "⚖️ Fairness Drift & Reranking",
    "📋 Batch Audit Log & Data Export"
])


# =============================================================================
# TAB 1: E-COMMERCE PRODUCT RECOMMENDATIONS
# =============================================================================
with tab_recs:
    st.markdown("### 🛍️ Personalized E-Commerce Recommendations")
    st.write("Inspect recommendations for any individual customer to observe how the **Base Continual Model** compares with the **Fairness-Aware Reranked Model** in real time.")

    # User Selection Row
    u_col1, u_col2 = st.columns([1, 2])
    with u_col1:
        user_options = sim.users_df["user_id"] + " - " + sim.users_df["name"] + " (" + sim.users_df["demographic_group"].str[:7] + ")"
        selected_option = st.selectbox("Select Customer to Inspect:", user_options, index=0)
        selected_uid = selected_option.split(" - ")[0]
        user_row = sim.users_df[sim.users_df["user_id"] == selected_uid].iloc[0]

    with u_col2:
        # User details card
        g_badge = "badge-group-a" if user_row["group_id"] == 0 else "badge-group-b"
        st.markdown(f"""
        <div style="background: #F1F5F9; border-radius: 8px; padding: 12px 16px; margin-top: 4px;">
            <b>Customer Profile:</b> {user_row['name']} (ID: <code>{user_row['user_id']}</code>)<br>
            <b>Group Assignment:</b> <span class="{g_badge}">{user_row['demographic_group']}</span> | 
            <b>Affinity Tier:</b> {user_row['activity_tier']} | 
            <b>Preferred Category:</b> {CATEGORY_ICONS.get(user_row['favorite_category'], '')} <b>{user_row['favorite_category']}</b>
        </div>
        """, unsafe_allow_html=True)

    # Compute recommendations for selected user
    raw_all = sim.recommender.rank_all_items(selected_uid, sim.products_df)
    raw_top_k = raw_all.head(sim.k)
    fair_top_k = sim.reranker.rerank_recommendations(selected_uid, user_row, raw_all, k=sim.k)

    # Side-by-side comparison of Base Model vs Fair Reranked Model
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("#### 🤖 Base Continual Model (Unmitigated)")
        st.caption("Standard online SGD matrix factorization updating solely on chronological clicks/purchases.")
        for idx, item in raw_top_k.reset_index(drop=True).iterrows():
            cat_icon = CATEGORY_ICONS.get(item["category"], "📦")
            is_fav = (item["category"] == user_row["favorite_category"])
            border_color = "#3B82F6" if is_fav else "#E2E8F0"
            st.markdown(f"""
            <div class="product-card" style="border-left: 4px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #1E293B;">#{idx+1} {item['title']}</span>
                    <span style="font-weight: 700; color: #059669;">${item['price']:.2f}</span>
                </div>
                <div style="margin-top: 4px; font-size: 0.85rem; color: #64748B;">
                    <span class="badge-cat">{cat_icon} {item['category']}</span> | Tag: <code>{item['tag']}</code>
                </div>
                <div style="margin-top: 6px; font-size: 0.85rem; color: #334155;">
                    Predicted Affinity Score: <b>{item['predicted_score']:.2f} / 5.0</b> 
                    {"<span style='color: #2563EB; font-weight:600;'> (Matches Preferred Category)</span>" if is_fav else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("#### ⚖️ Fairness-Aware Model (Reranked)")
        st.caption(f"Post-processing reranking applied with fairness weight α = {sim.reranking_alpha:.2f}.")
        for idx, item in fair_top_k.reset_index(drop=True).iterrows():
            cat_icon = CATEGORY_ICONS.get(item["category"], "📦")
            is_fav = (item["category"] == user_row["favorite_category"])
            border_color = "#10B981" if is_fav else "#E2E8F0"
            score_diff = item["reranked_score"] - item["raw_score"]
            diff_badge = f"<span style='color: #059669; font-weight: 600;'>+{score_diff:.2f}</span>" if score_diff > 0 else f"<span style='color: #64748B;'>{score_diff:.2f}</span>"
            st.markdown(f"""
            <div class="product-card" style="border-left: 4px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #1E293B;">#{idx+1} {item['title']}</span>
                    <span style="font-weight: 700; color: #059669;">${item['price']:.2f}</span>
                </div>
                <div style="margin-top: 4px; font-size: 0.85rem; color: #64748B;">
                    <span class="badge-cat">{cat_icon} {item['category']}</span> | Tag: <code>{item['tag']}</code>
                </div>
                <div style="margin-top: 6px; font-size: 0.85rem; color: #334155;">
                    Reranked Score: <b>{item['reranked_score']:.2f}</b> (Raw: {item['raw_score']:.2f}, Adj: {diff_badge})
                    {"<span style='color: #059669; font-weight:600;'> (Protected Alignment)</span>" if is_fav else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Customer's Historical Activity
    st.markdown("---")
    st.markdown("#### 📜 Customer Interaction History Across Retraining Rounds")
    if not sim.cumulative_interactions_df.empty:
        u_history = sim.cumulative_interactions_df[sim.cumulative_interactions_df["user_id"] == selected_uid]
        if not u_history.empty:
            merged_hist = u_history.merge(sim.products_df[["item_id", "title", "price"]], on="item_id", how="left")
            st.dataframe(
                merged_hist[["batch_id", "item_id", "title", "item_category", "rating", "is_positive"]].rename(
                    columns={
                        "batch_id": "Round",
                        "item_id": "Item ID",
                        "title": "Product Title",
                        "item_category": "Category",
                        "rating": "User Feedback Rating",
                        "is_positive": "Liked (≥3.8)"
                    }
                ),
                use_container_width=True
            )
        else:
            st.info("No interactions recorded yet for this customer in completed batches.")


# =============================================================================
# TAB 2: CONTINUAL LEARNING & FAIRNESS ANALYTICS
# =============================================================================
with tab_analytics:
    st.markdown("### 📈 Continual Learning Dynamics & Fairness Trajectory")
    st.write("Tracking recommendation performance and fairness drift progression across successive retraining batches.")

    history_df = sim.get_history_dataframe()
    if history_df.empty:
        st.warning("No rounds processed yet. Click 'Process Next Batch' in the sidebar.")
    else:
        # Row 1: Charts
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            st.markdown("##### 1. Recommendation Quality Progression Across Rounds")
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            ax1.plot(history_df["Round_Num"], history_df["Raw Precision@K"], marker="o", color="#2563EB", linewidth=2.2, label=f"Raw Precision@{sim.k}")
            ax1.plot(history_df["Round_Num"], history_df["Fair Precision@K"], marker="s", color="#10B981", linewidth=2.0, linestyle="--", label=f"Fair Precision@{sim.k}")
            ax1.plot(history_df["Round_Num"], history_df["Raw Recall@K"], marker="^", color="#8B5CF6", linewidth=1.8, label=f"Raw Recall@{sim.k}")
            ax1.set_xlabel("Continual Learning Round")
            ax1.set_ylabel("Metric Value")
            ax1.set_ylim(0, 1.0)
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper left", frameon=True)
            st.pyplot(fig1)

        with ch_col2:
            st.markdown("##### 2. Demographic Parity Difference (Disparity) Trajectory")
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            ax2.plot(history_df["Round_Num"], history_df["Raw DP Difference"], marker="o", color="#DC2626", linewidth=2.2, label="Raw Model Disparity")
            ax2.plot(history_df["Round_Num"], history_df["Fair DP Difference"], marker="s", color="#10B981", linewidth=2.2, label="Fair Model Disparity")
            ax2.axhline(y=sim.fairness_threshold, color="#EF4444", linestyle=":", linewidth=2, label=f"Tolerance Threshold ε ({sim.fairness_threshold:.2f})")
            ax2.set_xlabel("Continual Learning Round")
            ax2.set_ylabel("Disparity |Group B - Group A|")
            ax2.set_ylim(0, max(0.35, history_df["Raw DP Difference"].max() + 0.05))
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper left", frameon=True)
            st.pyplot(fig2)

        # Row 2: Charts
        ch_col3, ch_col4 = st.columns(2)

        with ch_col3:
            st.markdown("##### 3. Disparate Impact Ratio (80% Rule Compliance)")
            fig3, ax3 = plt.subplots(figsize=(6, 3.8))
            ax3.plot(history_df["Round_Num"], history_df["Raw Disparate Impact"], marker="o", color="#EA580C", linewidth=2.2, label="Raw Disparate Impact")
            ax3.plot(history_df["Round_Num"], history_df["Fair Disparate Impact"], marker="s", color="#059669", linewidth=2.2, label="Fair Reranked DI")
            ax3.axhline(y=sim.min_disparate_impact, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Legal 80% Rule ({sim.min_disparate_impact:.2f})")
            ax3.set_xlabel("Continual Learning Round")
            ax3.set_ylabel("Disparate Impact Ratio")
            ax3.set_ylim(0.3, 1.05)
            ax3.grid(True, linestyle=":", alpha=0.6)
            ax3.legend(loc="lower left", frameon=True)
            st.pyplot(fig3)

        with ch_col4:
            st.markdown("##### 4. Group A vs Group B Precision Progression")
            fig4, ax4 = plt.subplots(figsize=(6, 3.8))
            ax4.plot(history_df["Round_Num"], history_df["Raw Group A Precision"], marker="o", color="#3B82F6", linewidth=2, label="Group A (Raw)")
            ax4.plot(history_df["Round_Num"], history_df["Raw Group B Precision"], marker="o", color="#F59E0B", linewidth=2, label="Group B (Raw)")
            ax4.plot(history_df["Round_Num"], history_df["Fair Group A Precision"], marker="s", color="#10B981", linewidth=2, linestyle="--", label="Group A (Fair Reranked)")
            ax4.set_xlabel("Continual Learning Round")
            ax4.set_ylabel(f"Precision@{sim.k}")
            ax4.set_ylim(0, 1.0)
            ax4.grid(True, linestyle=":", alpha=0.6)
            ax4.legend(loc="upper left", frameon=True)
            st.pyplot(fig4)


# =============================================================================
# TAB 3: FAIRNESS DRIFT & MITIGATION DEEP DIVE
# =============================================================================
with tab_mitigation:
    st.markdown("### ⚖️ Theoretical Formulation & Fairness-Preserving Mechanism")
    
    st.markdown(r"""
    #### 1. The Catastrophic Forgetting of Fairness Problem in Continual Recommenders
    In continual learning for e-commerce, recommendation models update incrementally on batches of user interactions:
    $$\theta_{t} = \theta_{t-1} - \eta \cdot \nabla \mathcal{L}(B_t; \theta_{t-1})$$
    When customer engagement shifts (e.g. higher activity volume or dominant purchasing from **Group B**), the model's loss gradient updates disproportionately in favor of Group B items and preferences. This creates **algorithmic feedback loops** and **fairness drift**:
    - The model preserves or improves overall accuracy while quietly degrading recommendation quality and representation for **Group A** (catastrophic forgetting of fairness).
    """)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(r"""
        #### 2. Fairness Metrics Monitored
        - **Demographic Parity Difference ($\Delta_{\text{DP}}$)**:
          $$\Delta_{\text{DP}} = |P(\text{Rec} \ge \tau | G=B) - P(\text{Rec} \ge \tau | G=A)|$$
        - **Disparate Impact Ratio ($\text{DI}$)**:
          $$\text{DI} = \frac{\min(SR_A, SR_B)}{\max(SR_A, SR_B)}$$
          *Target:* $\text{DI} \ge 0.80$ (Four-fifths rule).
        - **Equal Opportunity Difference ($\Delta_{\text{EO}}$)**:
          $$\Delta_{\text{EO}} = |\text{TPR}_B - \text{TPR}_A| = |\text{Recall}_B - \text{Recall}_A|$$
        """)

    with m_col2:
        st.markdown(r"""
        #### 3. Fairness-Preserving Reranking Mechanism
        When $\Delta_{\text{DP}} > \epsilon$ or $\text{DI} < 0.80$, the system activates post-processing score calibration:
        $$S_{\text{fair}}(u, i) = (1 - \alpha) \cdot S_{\text{pred}}(u, i) + \alpha \cdot S_{\text{parity}}(u, i, G_u)$$
        - **$S_{\text{pred}}(u, i)$**: Relevance score predicted by continual matrix factorization.
        - **$S_{\text{parity}}(u, i, G_u)$**: Representation adjustment boosting preferred categories and catalog diversity for underrepresented groups.
        - **$\alpha$**: Configurable trade-off parameter controlling fairness intervention intensity.
        """)

    st.markdown("---")
    st.markdown("#### 🧪 Interactive 'What-If' Parameter Sensitivity")
    st.write("Examine how varying the fairness weight $\\alpha$ balances recommendation precision vs disparity on the current batch:")

    # Quick interactive alpha curve
    alphas = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
    sens_records = []
    sample_users = sim.users_df.head(40)
    for a in alphas:
        temp_reranker = from_scratch_reranker = type(sim.reranker)(alpha=a)
        temp_recs = {}
        for _, urow in sample_users.iterrows():
            uid = urow["user_id"]
            r_all = sim.recommender.rank_all_items(uid, sim.products_df)
            temp_recs[uid] = temp_reranker.rerank_recommendations(uid, urow, r_all, k=sim.k)
        q_eval = evaluate_recommendations_at_k(temp_recs, sim.cumulative_interactions_df, sample_users, k=sim.k)
        f_eval = sim.fairness_evaluator.evaluate_fairness(
            group_a_precision=q_eval["group_a"]["precision_at_k"],
            group_b_precision=q_eval["group_b"]["precision_at_k"],
            group_a_recall=q_eval["group_a"]["recall_at_k"],
            group_b_recall=q_eval["group_b"]["recall_at_k"],
            group_a_mean_score=3.5,
            group_b_mean_score=3.5
        )
        sens_records.append({
            "Fairness Weight α": a,
            "Precision@K": q_eval["overall"]["precision_at_k"],
            "Demographic Parity Diff": f_eval["demographic_parity_diff"],
            "Disparate Impact Ratio": f_eval["disparate_impact_ratio"]
        })
    sens_df = pd.DataFrame(sens_records)
    
    sens_c1, sens_c2 = st.columns([1, 1])
    with sens_c1:
        st.dataframe(sens_df, use_container_width=True)
    with sens_c2:
        fig_s, ax_s = plt.subplots(figsize=(6, 3.2))
        ax_s.plot(sens_df["Fairness Weight α"], sens_df["Demographic Parity Diff"], marker="o", color="#DC2626", label="Parity Disparity (Lower is Better)")
        ax_s.plot(sens_df["Fairness Weight α"], sens_df["Disparate Impact Ratio"], marker="s", color="#059669", label="Disparate Impact (Higher is Better)")
        ax_s.set_xlabel("Reranking Weight α")
        ax_s.set_ylabel("Fairness Metric Value")
        ax_s.grid(True, linestyle=":", alpha=0.6)
        ax_s.legend(loc="center right")
        st.pyplot(fig_s)


# =============================================================================
# TAB 4: AUDIT LOG & DATA EXPORT
# =============================================================================
with tab_audit:
    st.markdown("### 📋 Continual Retraining Audit Log")
    st.write("Complete audit records of continual learning rounds, accuracy metrics, and group fairness measurements across batches.")

    history_df = sim.get_history_dataframe()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)

        exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
        with exp_c1:
            csv_history = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Round History",
                data=csv_history,
                file_name="fairness_continual_learning_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_c2:
            csv_users = sim.users_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Users Table",
                data=csv_users,
                file_name="users_metadata.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_c3:
            csv_catalog = sim.products_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Product Catalog",
                data=csv_catalog,
                file_name="product_catalog.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_c4:
            csv_interactions = sim.cumulative_interactions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Interactions",
                data=csv_interactions,
                file_name="interactions_batches.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")
        # Direct ZIP package download
        zip_path = "data/fairness_continual_recommender_dataset.zip"
        if os.path.exists(zip_path):
            with open(zip_path, "rb") as zf:
                zip_bytes = zf.read()
            st.download_button(
                label="📦 Download Complete Dataset Package (.ZIP - Users, Catalog, Batches & Readme)",
                data=zip_bytes,
                file_name="fairness_continual_recommender_dataset.zip",
                mime="application/zip",
                use_container_width=True
            )
