"""
Continual Learning Recommender Module
Fairness-Aware Continual Learning Recommendation System

Implements incremental Matrix Factorization with online SGD updates
using the River machine learning library for streaming recommendation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from river import reco, optim


class ContinualRecommender:
    """
    Incremental Recommendation Engine using Online Latent Factor Matrix Factorization.
    Adapts continually as new batches of customer interactions arrive,
    without requiring retraining from scratch.
    """

    def __init__(self, n_factors: int = 10, lr: float = 0.025, l2: float = 1e-4, seed: int = 42):
        self.n_factors = n_factors
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.model = reco.BiasedMF(
            n_factors=self.n_factors,
            bias_optimizer=optim.SGD(self.lr),
            latent_optimizer=optim.SGD(self.lr),
            l2_bias=self.l2,
            l2_latent=self.l2,
            seed=self.seed
        )
        # Stateful tracking
        self.total_interactions: int = 0
        self.rounds_completed: int = 0
        self.user_history: Dict[str, List[str]] = {}
        self.user_category_counts: Dict[str, Dict[str, int]] = {}
        self.item_counts: Dict[str, int] = {}
        self.global_mean: float = 3.5

    def update_batch(self, batch_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Incrementally updates model weights using a batch of interactions.
        Simulates one continual learning round.
        """
        ratings = []
        for _, row in batch_df.iterrows():
            u_id = str(row["user_id"])
            i_id = str(row["item_id"])
            rating = float(row["rating"])
            cat = str(row.get("item_category", "Unknown"))
            
            # Online incremental update step
            self.model.learn_one(u_id, i_id, rating)
            
            # Update stateful metadata
            if u_id not in self.user_history:
                self.user_history[u_id] = []
                self.user_category_counts[u_id] = {}
            self.user_history[u_id].append(i_id)
            self.user_category_counts[u_id][cat] = self.user_category_counts[u_id].get(cat, 0) + 1
            self.item_counts[i_id] = self.item_counts.get(i_id, 0) + 1
            ratings.append(rating)

        self.total_interactions += len(batch_df)
        self.rounds_completed += 1
        if ratings:
            batch_mean = float(np.mean(ratings))
            # Exponential moving average of global rating
            self.global_mean = 0.8 * self.global_mean + 0.2 * batch_mean

        return {
            "round": self.rounds_completed,
            "batch_interactions": len(batch_df),
            "total_interactions": self.total_interactions,
            "unique_users_in_batch": batch_df["user_id"].nunique(),
            "batch_mean_rating": round(float(np.mean(ratings)), 2) if ratings else 0.0
        }

    def predict_score(self, user_id: str, item_id: str, item_category: Optional[str] = None) -> float:
        """
        Predicts relevance score for a given user-item pair.
        """
        score = self.model.predict_one(user_id, item_id)
        # If score is nan or None (unseen cold user/item), fall back to global mean
        if score is None or np.isnan(score):
            score = self.global_mean
            
        # Slight contextual boost if item matches user's preferred category
        if user_id in self.user_category_counts and item_category:
            user_cats = self.user_category_counts[user_id]
            total_u = sum(user_cats.values())
            if total_u > 0:
                cat_ratio = user_cats.get(item_category, 0) / total_u
                score += cat_ratio * 0.4
                
        # Clip score between 1.0 and 5.0
        return float(np.clip(score, 1.0, 5.0))

    def rank_all_items(
        self,
        user_id: str,
        products_df: pd.DataFrame,
        exclude_interacted: bool = False
    ) -> pd.DataFrame:
        """
        Scores and ranks all candidate items in the catalog for the specified user.
        """
        scores = []
        user_interacted = set(self.user_history.get(user_id, []))
        
        for _, prod in products_df.iterrows():
            i_id = prod["item_id"]
            if exclude_interacted and i_id in user_interacted:
                continue
            pred_score = self.predict_score(user_id, i_id, prod["category"])
            scores.append({
                "item_id": i_id,
                "title": prod["title"],
                "category": prod["category"],
                "price": prod["price"],
                "tag": prod["tag"],
                "predicted_score": round(pred_score, 3)
            })
            
        ranked_df = pd.DataFrame(scores)
        ranked_df = ranked_df.sort_values(by="predicted_score", ascending=False).reset_index(drop=True)
        return ranked_df

    def get_top_k(
        self,
        user_id: str,
        products_df: pd.DataFrame,
        k: int = 5,
        exclude_interacted: bool = False
    ) -> pd.DataFrame:
        """
        Returns top-k recommended items for a user.
        """
        ranked_df = self.rank_all_items(user_id, products_df, exclude_interacted=exclude_interacted)
        return ranked_df.head(k).copy()
