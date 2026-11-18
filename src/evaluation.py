"""
Evaluation Metrics Module
Fairness-Aware Continual Learning Recommendation System

Implements Precision@K, Recall@K, HitRate@K, and NDCG@K for ranking quality assessment.
Calculates overall and per-group recommendation performance across rounds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Optional


def compute_dcg(relevance_list: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at K."""
    relevance_list = relevance_list[:k]
    if not relevance_list:
        return 0.0
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_list))
    return float(dcg)


def compute_ndcg(recommended_items: List[str], ground_truth: Set[str], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at K."""
    if not ground_truth:
        return 0.0
    binary_rels = [1 if item in ground_truth else 0 for item in recommended_items[:k]]
    actual_dcg = compute_dcg(binary_rels, k)
    ideal_rels = sorted([1] * min(len(ground_truth), k) + [0] * max(0, k - len(ground_truth)), reverse=True)
    ideal_dcg = compute_dcg(ideal_rels, k)
    if ideal_dcg == 0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def evaluate_recommendations_at_k(
    user_recommendations: Dict[str, pd.DataFrame],
    interaction_df: pd.DataFrame,
    users_df: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 3.5
) -> Dict[str, Any]:
    """
    Computes Precision@K, Recall@K, HitRate@K, and NDCG@K
    both overall and broken down by protected group.
    """
    # Build user ground-truth relevant items
    rel_interactions = interaction_df[interaction_df["rating"] >= relevance_threshold]
    user_ground_truth: Dict[str, Set[str]] = (
        rel_interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    user_group_map = dict(zip(users_df["user_id"], users_df["group_id"]))

    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []

    group_metrics = {
        0: {"precisions": [], "recalls": [], "hit_rates": [], "ndcgs": []},
        1: {"precisions": [], "recalls": [], "hit_rates": [], "ndcgs": []}
    }

    for u_id, rec_df in user_recommendations.items():
        g_id = user_group_map.get(u_id, 0)
        gt_items = user_ground_truth.get(u_id, set())
        rec_list = rec_df["item_id"].head(k).tolist()

        if not gt_items:
            # If user has no historical positives yet, use category match proxy
            u_row = users_df[users_df["user_id"] == u_id]
            if not u_row.empty:
                fav_cat = u_row.iloc[0]["favorite_category"]
                cat_matches = (rec_df.head(k)["category"] == fav_cat).sum()
                prec = cat_matches / k
                rec = cat_matches / max(k, 1)
                hit = 1.0 if cat_matches > 0 else 0.0
                ndcg_val = prec
            else:
                prec, rec, hit, ndcg_val = 0.0, 0.0, 0.0, 0.0
        else:
            hits = len(set(rec_list).intersection(gt_items))
            prec = hits / k
            rec = hits / len(gt_items)
            hit = 1.0 if hits > 0 else 0.0
            ndcg_val = compute_ndcg(rec_list, gt_items, k)

        precisions.append(prec)
        recalls.append(rec)
        hit_rates.append(hit)
        ndcgs.append(ndcg_val)

        group_metrics[g_id]["precisions"].append(prec)
        group_metrics[g_id]["recalls"].append(rec)
        group_metrics[g_id]["hit_rates"].append(hit)
        group_metrics[g_id]["ndcgs"].append(ndcg_val)

    return {
        "k": k,
        "overall": {
            "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "hit_rate_at_k": round(float(np.mean(hit_rates)), 4) if hit_rates else 0.0,
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        },
        "group_a": {
            "name": "Group A (Standard Shoppers)",
            "precision_at_k": round(float(np.mean(group_metrics[0]["precisions"])), 4) if group_metrics[0]["precisions"] else 0.0,
            "recall_at_k": round(float(np.mean(group_metrics[0]["recalls"])), 4) if group_metrics[0]["recalls"] else 0.0,
            "hit_rate_at_k": round(float(np.mean(group_metrics[0]["hit_rates"])), 4) if group_metrics[0]["hit_rates"] else 0.0,
            "ndcg_at_k": round(float(np.mean(group_metrics[0]["ndcgs"])), 4) if group_metrics[0]["ndcgs"] else 0.0,
        },
        "group_b": {
            "name": "Group B (Premium Shoppers)",
            "precision_at_k": round(float(np.mean(group_metrics[1]["precisions"])), 4) if group_metrics[1]["precisions"] else 0.0,
            "recall_at_k": round(float(np.mean(group_metrics[1]["recalls"])), 4) if group_metrics[1]["recalls"] else 0.0,
            "hit_rate_at_k": round(float(np.mean(group_metrics[1]["hit_rates"])), 4) if group_metrics[1]["hit_rates"] else 0.0,
            "ndcg_at_k": round(float(np.mean(group_metrics[1]["ndcgs"])), 4) if group_metrics[1]["ndcgs"] else 0.0,
        }
    }
