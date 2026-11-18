"""
Continual Learning Simulation Manager
Fairness-Aware Continual Learning Recommendation System

Manages chronological batch progression, stateful model updating,
and records dual tracking (Raw Base Model vs Fairness-Reranked Model).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from src.data_generator import generate_synthetic_dataset
from src.recommender import ContinualRecommender
from src.fairness import FairnessEvaluator, FairnessReranker
from src.evaluation import evaluate_recommendations_at_k


class ContinualSimulationManager:
    """
    Coordinates chronological batch execution, online continual learning,
    and round-by-round fairness and quality monitoring.
    """

    def __init__(
        self,
        n_users: int = 100,
        n_batches: int = 6,
        k: int = 5,
        fairness_threshold: float = 0.12,
        min_disparate_impact: float = 0.80,
        reranking_alpha: float = 0.35,
        seed: int = 42
    ):
        self.n_users = n_users
        self.n_batches = n_batches
        self.k = k
        self.fairness_threshold = fairness_threshold
        self.min_disparate_impact = min_disparate_impact
        self.reranking_alpha = reranking_alpha
        self.seed = seed

        # Load synthetic e-commerce data
        self.users_df, self.products_df, self.batches = generate_synthetic_dataset(
            n_users=n_users, n_batches=n_batches, seed=seed
        )

        # Initialize recommender & evaluators
        self.recommender = ContinualRecommender(n_factors=10, lr=0.03, l2=1e-4, seed=seed)
        self.fairness_evaluator = FairnessEvaluator(
            fairness_threshold=fairness_threshold,
            min_disparate_impact=min_disparate_impact
        )
        self.reranker = FairnessReranker(alpha=reranking_alpha)

        # Simulation state
        self.current_batch_idx: int = 0
        self.cumulative_interactions_df = pd.DataFrame()
        self.round_history: List[Dict[str, Any]] = []

    def set_parameters(self, k: int, fairness_threshold: float, min_disparate_impact: float, reranking_alpha: float):
        """Allows dynamic adjustment of threshold and reranker weight from UI."""
        self.k = k
        self.fairness_threshold = fairness_threshold
        self.min_disparate_impact = min_disparate_impact
        self.reranking_alpha = reranking_alpha
        self.fairness_evaluator.fairness_threshold = fairness_threshold
        self.fairness_evaluator.min_disparate_impact = min_disparate_impact
        self.reranker.alpha = reranking_alpha

    def has_next_batch(self) -> bool:
        return self.current_batch_idx < len(self.batches)

    def process_next_batch(self) -> Optional[Dict[str, Any]]:
        """
        Advances one continual learning round:
        1. Ingests current batch interactions.
        2. Updates recommender incrementally.
        3. Computes top-k recommendations for all users (Raw and Reranked).
        4. Evaluates recommendation quality (Precision@K, Recall@K, etc.).
        5. Computes group fairness metrics and drift detection.
        6. Appends results to history.
        """
        if not self.has_next_batch():
            return None

        batch_df = self.batches[self.current_batch_idx]
        self.current_batch_idx += 1
        round_num = self.current_batch_idx

        # Incremental model update
        update_info = self.recommender.update_batch(batch_df)

        # Accumulate interactions
        self.cumulative_interactions_df = pd.concat(
            [self.cumulative_interactions_df, batch_df],
            ignore_index=True
        )

        # Generate predictions for all users
        raw_recs: Dict[str, pd.DataFrame] = {}
        fair_recs: Dict[str, pd.DataFrame] = {}
        raw_scores_g0, raw_scores_g1 = [], []
        fair_scores_g0, fair_scores_g1 = [], []

        for _, user_row in self.users_df.iterrows():
            u_id = user_row["user_id"]
            g_id = user_row["group_id"]
            
            # Raw recommendations
            raw_top = self.recommender.get_top_k(u_id, self.products_df, k=self.k)
            raw_recs[u_id] = raw_top
            mean_s = float(raw_top["predicted_score"].mean()) if not raw_top.empty else 0.0
            if g_id == 0:
                raw_scores_g0.append(mean_s)
            else:
                raw_scores_g1.append(mean_s)

            # Fair reranked recommendations
            raw_all = self.recommender.rank_all_items(u_id, self.products_df)
            fair_top = self.reranker.rerank_recommendations(u_id, user_row, raw_all, k=self.k)
            fair_recs[u_id] = fair_top
            mean_fs = float(fair_top["reranked_score"].mean()) if not fair_top.empty else 0.0
            if g_id == 0:
                fair_scores_g0.append(mean_fs)
            else:
                fair_scores_g1.append(mean_fs)

        # Evaluate Raw Quality
        raw_quality = evaluate_recommendations_at_k(
            raw_recs, self.cumulative_interactions_df, self.users_df, k=self.k
        )
        raw_fairness = self.fairness_evaluator.evaluate_fairness(
            group_a_precision=raw_quality["group_a"]["precision_at_k"],
            group_b_precision=raw_quality["group_b"]["precision_at_k"],
            group_a_recall=raw_quality["group_a"]["recall_at_k"],
            group_b_recall=raw_quality["group_b"]["recall_at_k"],
            group_a_mean_score=float(np.mean(raw_scores_g0)) if raw_scores_g0 else 0.0,
            group_b_mean_score=float(np.mean(raw_scores_g1)) if raw_scores_g1 else 0.0,
        )

        # Evaluate Fair Reranked Quality
        fair_quality = evaluate_recommendations_at_k(
            fair_recs, self.cumulative_interactions_df, self.users_df, k=self.k
        )
        fair_fairness = self.fairness_evaluator.evaluate_fairness(
            group_a_precision=fair_quality["group_a"]["precision_at_k"],
            group_b_precision=fair_quality["group_b"]["precision_at_k"],
            group_a_recall=fair_quality["group_a"]["recall_at_k"],
            group_b_recall=fair_quality["group_b"]["recall_at_k"],
            group_a_mean_score=float(np.mean(fair_scores_g0)) if fair_scores_g0 else 0.0,
            group_b_mean_score=float(np.mean(fair_scores_g1)) if fair_scores_g1 else 0.0,
        )

        snapshot = {
            "round": round_num,
            "batch_size": len(batch_df),
            "total_interactions": len(self.cumulative_interactions_df),
            "batch_users": batch_df["user_id"].nunique(),
            "raw_quality": raw_quality,
            "raw_fairness": raw_fairness,
            "fair_quality": fair_quality,
            "fair_fairness": fair_fairness,
            "raw_recs_sample": {uid: raw_recs[uid] for uid in list(raw_recs.keys())[:12]},
            "fair_recs_sample": {uid: fair_recs[uid] for uid in list(fair_recs.keys())[:12]},
        }

        self.round_history.append(snapshot)
        return snapshot

    def run_all_batches(self):
        """Runs all remaining batches in order."""
        while self.has_next_batch():
            self.process_next_batch()

    def get_history_dataframe(self) -> pd.DataFrame:
        """Flattens round history for easy visualization in Streamlit charts."""
        records = []
        for s in self.round_history:
            r = s["round"]
            rq = s["raw_quality"]["overall"]
            rq_a = s["raw_quality"]["group_a"]
            rq_b = s["raw_quality"]["group_b"]
            rf = s["raw_fairness"]

            fq = s["fair_quality"]["overall"]
            fq_a = s["fair_quality"]["group_a"]
            fq_b = s["fair_quality"]["group_b"]
            ff = s["fair_fairness"]

            records.append({
                "Round": f"Round {r}",
                "Round_Num": r,
                "Total Interactions": s["total_interactions"],
                # Raw model
                "Raw Precision@K": rq["precision_at_k"],
                "Raw Recall@K": rq["recall_at_k"],
                "Raw NDCG@K": rq["ndcg_at_k"],
                "Raw Group A Precision": rq_a["precision_at_k"],
                "Raw Group B Precision": rq_b["precision_at_k"],
                "Raw DP Difference": rf["demographic_parity_diff"],
                "Raw Disparate Impact": rf["disparate_impact_ratio"],
                "Raw EO Difference": rf["equal_opportunity_diff"],
                "Raw Drift Alert": rf["is_violated"],
                # Fair model
                "Fair Precision@K": fq["precision_at_k"],
                "Fair Recall@K": fq["recall_at_k"],
                "Fair NDCG@K": fq["ndcg_at_k"],
                "Fair Group A Precision": fq_a["precision_at_k"],
                "Fair Group B Precision": fq_b["precision_at_k"],
                "Fair DP Difference": ff["demographic_parity_diff"],
                "Fair Disparate Impact": ff["disparate_impact_ratio"],
                "Fair EO Difference": ff["equal_opportunity_diff"],
                "Fair Drift Alert": ff["is_violated"],
                "Threshold": self.fairness_threshold,
                "Min DI Threshold": self.min_disparate_impact
            })
        return pd.DataFrame(records)
