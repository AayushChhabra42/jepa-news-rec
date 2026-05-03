"""XGBoost LambdaRank baseline with hand-engineered features."""

import os
import sys
import pickle
import argparse
import logging
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import normalize
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def compute_user_profiles(
    user_sequences: dict,
    cat_ids: np.ndarray,
    text_embeddings: np.ndarray,
    num_categories: int,
) -> dict[str, dict]:
    """Compute per-user feature profiles from click history.

    Features:
      - click_rate_per_cat: (18,) — fraction of clicks in each category
      - avg_recency: mean recency of clicks (position from end)
      - num_clicks: total number of clicks
      - mean_embedding: (384,) — mean of clicked article embeddings
      - top_categories: set of top-3 clicked categories
    """
    profiles = {}

    for uid, data in user_sequences.items():
        history = data["history_ids"]
        if not history:
            continue

        # Category distribution
        cat_counts = np.zeros(num_categories)
        for hid in history:
            cat_counts[cat_ids[hid]] += 1
        total = cat_counts.sum()
        click_rate = cat_counts / total if total > 0 else cat_counts

        # Top 3 categories
        top_cats = set(np.argsort(-cat_counts)[:3])

        # Mean embedding
        valid_ids = [h for h in history if h < len(text_embeddings)]
        mean_emb = text_embeddings[valid_ids].mean(axis=0) if valid_ids else np.zeros(text_embeddings.shape[1])

        profiles[uid] = {
            "click_rate_per_cat": click_rate,
            "num_clicks": len(history),
            "mean_embedding": mean_emb,
            "top_categories": top_cats,
        }

    return profiles


def compute_global_ctr(behaviors: list[dict], news_id2idx: dict) -> dict[int, float]:
    """Compute global CTR for each article in the training window."""
    impressions = defaultdict(lambda: {"shown": 0, "clicked": 0})

    for record in behaviors:
        for nid, label in record["impressions"]:
            if nid in news_id2idx:
                idx = news_id2idx[nid]
                impressions[idx]["shown"] += 1
                if label == 1:
                    impressions[idx]["clicked"] += 1

    ctr = {}
    for idx, counts in impressions.items():
        ctr[idx] = counts["clicked"] / counts["shown"] if counts["shown"] > 0 else 0.0

    return ctr


def build_features(
    user_sequences: dict,
    user_profiles: dict,
    global_ctr: dict,
    cat_ids: np.ndarray,
    text_embeddings: np.ndarray,
    num_categories: int,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix for XGBRanker.

    Features per (user, candidate):
      1-18: User click rate per category (18 features)
      19: User number of clicks
      20: Candidate global CTR
      21: Cosine similarity (user mean emb ↔ candidate emb)
      22: Category match flag (candidate cat ∈ user top-3)
      23: Candidate has entity flag
      24: Impression position (0-indexed)

    Returns:
        X: (num_samples, num_features)
        y: (num_samples,) — binary labels
        groups: list[int] — group sizes for LambdaRank
    """
    X_list = []
    y_list = []
    groups = []

    for uid, data in tqdm(user_sequences.items(), desc=f"Building {split} features"):
        if uid not in user_profiles:
            continue
        profile = user_profiles[uid]

        for impression in data["impressions"]:
            candidates = impression["candidates"]
            labels = impression["labels"]

            if not candidates:
                continue

            group_features = []
            group_labels = []

            for pos, (cand_id, label) in enumerate(zip(candidates, labels)):
                feat = np.zeros(24)

                # 1-18: User click rate per category
                feat[:num_categories] = profile["click_rate_per_cat"][:num_categories]

                # 19: User total clicks
                feat[18] = profile["num_clicks"]

                # 20: Candidate global CTR
                feat[19] = global_ctr.get(cand_id, 0.0)

                # 21: Cosine similarity
                if cand_id < len(text_embeddings):
                    cand_emb = text_embeddings[cand_id]
                    user_emb = profile["mean_embedding"]
                    norm_c = np.linalg.norm(cand_emb)
                    norm_u = np.linalg.norm(user_emb)
                    if norm_c > 0 and norm_u > 0:
                        feat[20] = np.dot(user_emb, cand_emb) / (norm_u * norm_c)

                # 22: Category match
                cand_cat = cat_ids[cand_id] if cand_id < len(cat_ids) else 0
                feat[21] = 1.0 if cand_cat in profile["top_categories"] else 0.0

                # 23: Entity flag — reuse from cat/subcat pipeline
                feat[22] = 1.0  # Placeholder — actual flag from preprocessing

                # 24: Position in impression list
                feat[23] = pos

                group_features.append(feat)
                group_labels.append(label)

            if group_features:
                X_list.extend(group_features)
                y_list.extend(group_labels)
                groups.append(len(group_features))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"  {split}: {X.shape[0]} samples, {len(groups)} groups")
    return X, y, groups


def train_xgb(
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    n_estimators: int = 500,
    max_depth: int = 6,
    eta: float = 0.1,
    device: str = "cpu",
):
    """Train XGBRanker baseline."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data
    print("Loading preprocessed data...")
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    text_embeddings = data["text_embeddings"]
    cat_ids = data["article_features"]["cat_ids"]
    vocabs = data["vocabs"]
    train_users_all = data["train_users"]
    
    # Create a dev split from train users (last 10%)
    user_ids = list(train_users_all.keys())
    split_idx = int(len(user_ids) * 0.9)
    train_users = {uid: train_users_all[uid] for uid in user_ids[:split_idx]}
    dev_users = {uid: train_users_all[uid] for uid in user_ids[split_idx:]}

    # Re-parse behaviors for CTR computation
    from data.preprocess import parse_behaviors_tsv
    raw_dir = processed_dir.replace("processed", "raw")
    dataset_name = "mind-small"  # TODO: read from config
    train_behaviors = parse_behaviors_tsv(
        os.path.join(raw_dir, dataset_name, "train", "behaviors.tsv")
    )

    num_categories = len(vocabs["cat2idx"])

    # Compute features
    print("\nComputing user profiles...")
    user_profiles = compute_user_profiles(
        train_users, cat_ids, text_embeddings, num_categories
    )
    # Also compute for dev users
    dev_profiles = compute_user_profiles(
        dev_users, cat_ids, text_embeddings, num_categories
    )
    all_profiles = {**user_profiles, **dev_profiles}

    print("Computing global CTR...")
    global_ctr = compute_global_ctr(train_behaviors, vocabs["news_id2idx"])

    print("\nBuilding feature matrices...")
    X_train, y_train, groups_train = build_features(
        train_users, all_profiles, global_ctr, cat_ids,
        text_embeddings, num_categories, split="train"
    )
    X_dev, y_dev, groups_dev = build_features(
        dev_users, all_profiles, global_ctr, cat_ids,
        text_embeddings, num_categories, split="dev"
    )

    # Train XGBRanker
    print(f"\nTraining XGBRanker (n_estimators={n_estimators}, max_depth={max_depth})")
    print("=" * 60)

    ranker = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=eta,
        tree_method="hist",
        eval_metric="ndcg@5",
        early_stopping_rounds=50,
    )

    ranker.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_dev, y_dev)],
        eval_group=[groups_dev],
        verbose=True,
    )

    # Save model
    model_path = os.path.join(checkpoint_dir, "xgb_ranker.json")
    ranker.save_model(model_path)
    print(f"\n✓ XGBRanker saved to {model_path}")

    # Feature importance
    importance = ranker.feature_importances_
    feature_names = [
        *[f"cat_rate_{i}" for i in range(num_categories)],
        "num_clicks", "global_ctr", "cosine_sim",
        "cat_match", "entity_flag", "position",
    ]
    print("\nFeature importance (top 10):")
    sorted_idx = np.argsort(-importance)
    for i in sorted_idx[:10]:
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"  {name}: {importance[i]:.4f}")

    return ranker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train XGBoost baseline")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    train_xgb(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
