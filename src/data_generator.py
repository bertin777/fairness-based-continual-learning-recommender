"""
Data Generation & Preprocessing Module
Fairness-Aware Continual Learning Recommendation System

NOTE: This module generates a reproducible synthetic e-commerce interaction dataset
specifically designed for demonstrating continual learning and fairness drift across batches.
It does NOT claim to be or use real Amazon customer data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


CATEGORIES = [
    "Electronics",
    "Fashion & Apparel",
    "Home & Kitchen",
    "Beauty & Care",
    "Books & Media"
]

CATEGORY_ICONS = {
    "Electronics": "⚡",
    "Fashion & Apparel": "👗",
    "Home & Kitchen": "🍳",
    "Beauty & Care": "✨",
    "Books & Media": "📚"
}

# 40 Products across categories
PRODUCTS_CATALOG = [
    # Electronics
    {"item_id": "E101", "title": "Wireless Noise-Canceling Headphones", "category": "Electronics", "price": 129.99, "tag": "Audio"},
    {"item_id": "E102", "title": "Smart Fitness Watch V2", "category": "Electronics", "price": 89.99, "tag": "Wearables"},
    {"item_id": "E103", "title": "Ultra-HD 4K Action Camera", "category": "Electronics", "price": 199.99, "tag": "Cameras"},
    {"item_id": "E104", "title": "Mechanical RGB Gaming Keyboard", "category": "Electronics", "price": 69.99, "tag": "Gaming"},
    {"item_id": "E105", "title": "Fast Wireless Charging Dock 3-in-1", "category": "Electronics", "price": 39.99, "tag": "Accessories"},
    {"item_id": "E106", "title": "Portable Bluetooth Speaker (Waterproof)", "category": "Electronics", "price": 49.99, "tag": "Audio"},
    {"item_id": "E107", "title": "Ergonomic Vertical Mouse", "category": "Electronics", "price": 34.99, "tag": "Accessories"},
    {"item_id": "E108", "title": "USB-C Dual Display Hub Pro", "category": "Electronics", "price": 54.99, "tag": "Computers"},

    # Fashion & Apparel
    {"item_id": "F201", "title": "Classic Organic Cotton Hoodie", "category": "Fashion & Apparel", "price": 45.00, "tag": "Casual"},
    {"item_id": "F202", "title": "Lightweight Running Sneakers", "category": "Fashion & Apparel", "price": 79.99, "tag": "Footwear"},
    {"item_id": "F203", "title": "Water-Resistant Commuter Backpack", "category": "Fashion & Apparel", "price": 59.99, "tag": "Bags"},
    {"item_id": "F204", "title": "UV-Protection Polarized Sunglasses", "category": "Fashion & Apparel", "price": 28.50, "tag": "Eyewear"},
    {"item_id": "F205", "title": "Slim Fit Breathable Chinos", "category": "Fashion & Apparel", "price": 49.00, "tag": "Pants"},
    {"item_id": "F206", "title": "Seamless Moisture-Wicking T-Shirt", "category": "Fashion & Apparel", "price": 22.00, "tag": "Athletic"},
    {"item_id": "F207", "title": "Merino Wool Winter Scarf", "category": "Fashion & Apparel", "price": 32.00, "tag": "Accessories"},
    {"item_id": "F208", "title": "Minimalist Leather Cardholder", "category": "Fashion & Apparel", "price": 19.99, "tag": "Accessories"},

    # Home & Kitchen
    {"item_id": "H301", "title": "Stainless Steel Espresso Machine", "category": "Home & Kitchen", "price": 149.00, "tag": "Appliances"},
    {"item_id": "H302", "title": "Digital Air Fryer 5.8-Quart", "category": "Home & Kitchen", "price": 89.99, "tag": "Appliances"},
    {"item_id": "H303", "title": "Cast Iron Pre-Seasoned Skillet", "category": "Home & Kitchen", "price": 39.99, "tag": "Cookware"},
    {"item_id": "H304", "title": "Aroma Essential Oil Diffuser", "category": "Home & Kitchen", "price": 25.99, "tag": "Wellness"},
    {"item_id": "H305", "title": "Insulated Stainless Water Bottle 1L", "category": "Home & Kitchen", "price": 24.50, "tag": "Dining"},
    {"item_id": "H306", "title": "Memory Foam Ergonomic Contour Pillow", "category": "Home & Kitchen", "price": 42.00, "tag": "Bedding"},
    {"item_id": "H307", "title": "Bamboo Cutting Board with Juice Groove", "category": "Home & Kitchen", "price": 21.99, "tag": "Kitchenware"},
    {"item_id": "H308", "title": "Smart LED Dimmable Desk Lamp", "category": "Home & Kitchen", "price": 35.00, "tag": "Lighting"},

    # Beauty & Care
    {"item_id": "B401", "title": "Hydrating Hyaluronic Acid Serum", "category": "Beauty & Care", "price": 26.00, "tag": "Skincare"},
    {"item_id": "B402", "title": "Sonic Electric Toothbrush Rechargeable", "category": "Beauty & Care", "price": 45.00, "tag": "Oral Care"},
    {"item_id": "B403", "title": "Organic Argan Oil Hair Mask", "category": "Beauty & Care", "price": 18.50, "tag": "Haircare"},
    {"item_id": "B404", "title": "Broad Spectrum Mineral Sunscreen SPF 50", "category": "Beauty & Care", "price": 21.00, "tag": "Skincare"},
    {"item_id": "B405", "title": "Vitamin C Radiance Face Cleanser", "category": "Beauty & Care", "price": 16.50, "tag": "Skincare"},
    {"item_id": "B406", "title": "Aromatherapy Lavender Bath Salts", "category": "Beauty & Care", "price": 14.99, "tag": "Body"},
    {"item_id": "B407", "title": "Compact Ionic Travel Hair Dryer", "category": "Beauty & Care", "price": 38.00, "tag": "Styling"},
    {"item_id": "B408", "title": "Nourishing Shea Butter Hand Cream", "category": "Beauty & Care", "price": 11.50, "tag": "Body"},

    # Books & Media
    {"item_id": "M501", "title": "Continual Machine Learning in Practice", "category": "Books & Media", "price": 48.00, "tag": "Tech Books"},
    {"item_id": "M502", "title": "Designing Data-Intensive Applications", "category": "Books & Media", "price": 52.00, "tag": "Tech Books"},
    {"item_id": "M503", "title": "Atomic Habits: Proven Framework", "category": "Books & Media", "price": 18.00, "tag": "Self-Help"},
    {"item_id": "M504", "title": "Deep Work: Rules for Focused Success", "category": "Books & Media", "price": 17.50, "tag": "Self-Help"},
    {"item_id": "M505", "title": "The Psychology of Money", "category": "Books & Media", "price": 19.99, "tag": "Finance"},
    {"item_id": "M506", "title": "Project Hail Mary: A Sci-Fi Odyssey", "category": "Books & Media", "price": 16.00, "tag": "Fiction"},
    {"item_id": "M507", "title": "Zero to One: Notes on Startups", "category": "Books & Media", "price": 22.00, "tag": "Business"},
    {"item_id": "M508", "title": "Clean Code: A Handbook of Agile Craft", "category": "Books & Media", "price": 44.00, "tag": "Tech Books"}
]


def generate_synthetic_dataset(n_users: int = 100, n_batches: int = 6, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    """
    Generates synthetic users, products catalog, and chronological interaction batches.
    
    Demonstration Groups:
      - Group A (Group 0): Standard / Emerging Shoppers (50% of users)
      - Group B (Group 1): Premium / Core Shoppers (50% of users)
      
    Simulated Continual Drift:
      - Round 1-2: Balanced activity across both groups.
      - Round 3-4: Popularity bias drift. Interaction frequency of Group B surges,
                   leading standard continual learning models to overfit Group B preferences
                   and degrade recommendations for Group A (fairness violation).
      - Round 5-6: Preference shift & catalog expansion drift.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate Users
    user_list = []
    first_names_a = ["Aarav", "Meera", "Rohan", "Priya", "Vikram", "Sneha", "Kiran", "Divya", "Arjun", "Ananya"]
    first_names_b = ["David", "Emma", "Liam", "Olivia", "Noah", "Sophia", "James", "Isabella", "Lucas", "Mia"]
    
    for uid in range(1, n_users + 1):
        # 50% Group A, 50% Group B
        group_id = 0 if uid <= n_users // 2 else 1
        group_label = "Group A (Standard Shoppers)" if group_id == 0 else "Group B (Premium Shoppers)"
        
        name_seed = (uid - 1) % len(first_names_a)
        name = f"{first_names_a[name_seed]}_{uid}" if group_id == 0 else f"{first_names_b[name_seed]}_{uid}"
        
        # Inherent category affinities
        fav_cat = rng.choice(CATEGORIES)
        user_list.append({
            "user_id": f"U{uid:03d}",
            "name": name,
            "demographic_group": group_label,
            "group_id": group_id,  # 0 or 1
            "favorite_category": fav_cat,
            "activity_tier": "Standard" if group_id == 0 else "Premium"
        })
        
    users_df = pd.DataFrame(user_list)
    products_df = pd.DataFrame(PRODUCTS_CATALOG)
    
    # 2. Generate Chronological Batches (Rounds 1 to n_batches)
    batches = []
    all_item_ids = products_df["item_id"].tolist()
    
    for batch_num in range(1, n_batches + 1):
        batch_records = []
        
        # Simulate drift in sampling probability
        # In batch 1, 2: balanced
        # In batch 3, 4, 5, 6: Group B interactions increase, creating algorithmic bias
        p_group_b = 0.50 if batch_num <= 2 else min(0.80, 0.50 + 0.08 * (batch_num - 2))
        
        n_interactions = 140 + batch_num * 20
        for _ in range(n_interactions):
            # Pick user according to group distribution
            is_group_b = rng.random() < p_group_b
            if is_group_b:
                user_row = users_df[users_df["group_id"] == 1].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
            else:
                user_row = users_df[users_df["group_id"] == 0].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
            
            u_id = user_row["user_id"]
            u_fav = user_row["favorite_category"]
            g_id = user_row["group_id"]
            
            # 60% chance user picks from their favorite category, 40% random
            if rng.random() < 0.65:
                cat_items = products_df[products_df["category"] == u_fav]["item_id"].tolist()
                item_id = rng.choice(cat_items)
            else:
                item_id = rng.choice(all_item_ids)
                
            item_row = products_df[products_df["item_id"] == item_id].iloc[0]
            
            # Feedback rating / relevance (1 to 5 stars, where >=4 is positive interaction)
            base_score = 4.2 if item_row["category"] == u_fav else 3.2
            # Add synthetic noise
            score = float(np.clip(rng.normal(base_score, 0.7), 1.0, 5.0))
            is_positive = 1 if score >= 3.8 else 0
            
            batch_records.append({
                "batch_id": batch_num,
                "user_id": u_id,
                "item_id": item_id,
                "rating": round(score, 1),
                "is_positive": is_positive,
                "group_id": g_id,
                "group_label": user_row["demographic_group"],
                "item_category": item_row["category"]
            })
            
        b_df = pd.DataFrame(batch_records)
        batches.append(b_df)
        
    return users_df, products_df, batches
