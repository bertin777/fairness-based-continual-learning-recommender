"""
Fairness Monitoring & Fairness-Preserving Reranking Module
Fairness-Aware Continual Learning Recommendation System

Implements:
1. Demographic Parity (Difference & Disparate Impact Ratio based on recommendation selection/precision rates)
2. Equal Opportunity Difference (True Positive Rate / Recall@K disparity across groups)
3. Fairness Drift Detection against configurable threshold
4. Fairness-Preserving Reranking correction mechanism (Post-Processing)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


class FairnessEvaluator:
    """
    Evaluates group fairness metrics across continual learning retraining rounds.
    Monitors Demographic Parity, Equal Opportunity, and detects fairness drift.
    """

    def __init__(self, fairness_threshold: float = 0.12, min_disparate_impact: float = 0.80):
        self.fairness_threshold = fairness_threshold  # Max allowed difference (e.g. 0.12)
        self.min_disparate_impact = min_disparate_impact  # Min acceptable ratio (80% four-fifths rule)

    def evaluate_fairness(
        self,
        group_a_precision: float,
        group_b_precision: float,
        group_a_recall: float,
        group_b_recall: float,
        group_a_mean_score: float,
        group_b_mean_score: float
    ) -> Dict[str, Any]:
        """
        Computes formal fairness metrics directly from group performance:
        - Demographic Parity Difference = |P_B - P_A|
        - Disparate Impact Ratio = min(P_A, P_B) / max(P_A, P_B)
        - Equal Opportunity Difference = |Recall_B - Recall_A|
        - Score Gap = |Score_B - Score_A|
        """
        # Demographic Parity Difference (Precision / Selection rate gap)
        dp_diff = abs(group_b_precision - group_a_precision)

        # Disparate Impact Ratio (80% rule: min / max)
        max_p = max(group_a_precision, group_b_precision, 1e-4)
        min_p = min(group_a_precision, group_b_precision)
        disparate_impact = min_p / max_p

        # Equal Opportunity Difference (True Positive Rate / Recall disparity)
        eo_diff = abs(group_b_recall - group_a_recall)

        # Score Parity Gap
        score_gap = abs(group_b_mean_score - group_a_mean_score)

        # Violation check
        is_violated = (dp_diff > self.fairness_threshold) or (disparate_impact < self.min_disparate_impact)

        return {
            "group_a_mean_score": round(group_a_mean_score, 3),
            "group_b_mean_score": round(group_b_mean_score, 3),
            "score_gap": round(score_gap, 3),
            "group_a_selection_rate": round(group_a_precision, 3),
            "group_b_selection_rate": round(group_b_precision, 3),
            "demographic_parity_diff": round(dp_diff, 3),
            "disparate_impact_ratio": round(disparate_impact, 3),
            "group_a_recall": round(group_a_recall, 3),
            "group_b_recall": round(group_b_recall, 3),
            "equal_opportunity_diff": round(eo_diff, 3),
            "threshold_configured": self.fairness_threshold,
            "min_disparate_impact_threshold": self.min_disparate_impact,
            "is_violated": is_violated,
            "status": "Fairness Drift Alert!" if is_violated else "Fairness Compliant"
        }


class FairnessReranker:
    """
    Fairness-Preserving Mechanism (Post-Processing Reranking).
    
    When continual learning introduces bias/drift against Group A,
    the reranker recalibrates candidate rankings by balancing:
      S_fair(u, i) = (1 - alpha) * S_pred(u, i) + alpha * S_fairness_boost(u, i, G_u)
    
    This counteracts continual forgetting of Group A and eliminates popularity bias.
    """

    def __init__(self, alpha: float = 0.35):
        """
        alpha: Weight given to fairness correction (0.0 = raw model, 1.0 = maximum parity boost).
        """
        self.alpha = alpha

    def rerank_recommendations(
        self,
        user_id: str,
        user_row: pd.Series,
        raw_ranked_df: pd.DataFrame,
        k: int = 5
    ) -> pd.DataFrame:
        """
        Reranks raw candidate scores to produce fair top-K recommendations.
        """
        g_id = int(user_row.get("group_id", 0))
        fav_cat = user_row.get("favorite_category", "")

        reranked_items = []
        for _, row in raw_ranked_df.iterrows():
            pred = float(row["predicted_score"])
            cat = row["category"]

            # Fairness bonus: If user belongs to disadvantaged group (Group A),
            # boost items matching their inherent category preference and counter
            # the continual drift caused by Group B dominance.
            if g_id == 0:
                is_fav = 1.0 if cat == fav_cat else 0.25
                fairness_component = 3.8 + is_fav * 1.2
            else:
                # Group B already has strong alignment
                fairness_component = pred

            # Blended score
            final_score = (1.0 - self.alpha) * pred + self.alpha * fairness_component

            item_dict = dict(row)
            item_dict["raw_score"] = round(pred, 3)
            item_dict["reranked_score"] = round(final_score, 3)
            reranked_items.append(item_dict)

        reranked_df = pd.DataFrame(reranked_items)
        reranked_df = reranked_df.sort_values(by="reranked_score", ascending=False).reset_index(drop=True)
        return reranked_df.head(k).copy()
