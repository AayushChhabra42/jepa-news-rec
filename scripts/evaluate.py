"""Stage 6: Evaluation — compare JEPA and XGBoost on validation set."""

import os
import sys
import pickle
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.item_encoder import ItemEncoder
from models.jepa import JEPA
from models.ranking_head import RankingHead, FineTuneModel
from scripts.finetune import ImpressionDataset, collate_impressions
from evaluation.metrics import evaluate_all, print_results
from baselines.xgboost_ranker import compute_global_ctr
from data.preprocess import parse_behaviors_tsv

logger = logging.getLogger(__name__)


def evaluate_jepa(
    processed_dir: str,
    checkpoint_path: str,
    max_seq_len: int = 50,
    batch_size: int = 128,
    scoring_mode: str = "mlp",
    predictor_type: str = "transformer",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Evaluate fine-tuned JEPA model on dev set."""
    print("\n" + "=" * 60)
    print("  Evaluating JEPA model")
    print("=" * 60)

    # Load data
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    text_embeddings = data["text_embeddings"]
    cat_ids_np = data["article_features"]["cat_ids"]
    subcat_ids_np = data["article_features"]["subcat_ids"]
    entity_flags_np = data["article_features"]["entity_flags"]
    vocabs = data["vocabs"]
    train_users_all = data["train_users"]
    user_ids = list(train_users_all.keys())
    split_idx = int(len(user_ids) * 0.9)
    dev_users = {uid: train_users_all[uid] for uid in user_ids[split_idx:]}

    cat_ids = torch.tensor(cat_ids_np, dtype=torch.long, device=device)
    subcat_ids = torch.tensor(subcat_ids_np, dtype=torch.long, device=device)
    entity_flags = torch.tensor(entity_flags_np, dtype=torch.float, device=device)

    # Compute Global CTR
    print("Computing global CTR...")
    raw_dir = processed_dir.replace("processed", "raw")
    dataset_name = "mind-small"
    if "mind-large" in processed_dir: dataset_name = "mind-large"
    train_behaviors = parse_behaviors_tsv(
        os.path.join(raw_dir, dataset_name, "train", "behaviors.tsv")
    )
    global_ctr_dict = compute_global_ctr(train_behaviors, vocabs["news_id2idx"])
    global_ctr_np = np.zeros(len(vocabs["news_id2idx"]), dtype=np.float32)
    for idx, ctr in global_ctr_dict.items():
        global_ctr_np[idx] = ctr
    global_ctr_tensor = torch.tensor(global_ctr_np, dtype=torch.float, device=device)

    # Build model
    item_encoder = ItemEncoder(
        text_embeddings=text_embeddings,
        num_categories=len(vocabs["cat2idx"]),
        num_subcategories=len(vocabs["subcat2idx"]),
    )
    jepa_model = JEPA(
        item_encoder=item_encoder,
        context_encoder_cfg={
            "d_model": 128, "nhead": 4, "num_layers": 4,
            "d_ff": 512, "dropout": 0.0, "max_seq_len": max_seq_len,
        },
        predictor_cfg={
            "type": predictor_type,
            "d_model": 128, "nhead": 4, "num_layers": 2,
            "d_ff": 256, "dropout": 0.0, "max_target_len": max_seq_len,
        },
        ema_cfg={"tau_start": 0.996, "tau_end": 0.9999, "total_steps": 1},
    )
    ranking_head = RankingHead(d_model=128, hidden_dim=256, dropout=0.0, mode=scoring_mode)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    jepa_model.load_state_dict(ckpt["jepa_state"])
    ranking_head.load_state_dict(ckpt["ranking_head_state"])

    ft_model = FineTuneModel(jepa_model, ranking_head).to(device)
    ft_model.eval()

    # Evaluate on dev set
    dataset = ImpressionDataset(dev_users, max_seq_len=max_seq_len, neg_sample_ratio=999)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate_impressions(b, max_seq_len),
    )

    all_impressions = []
    history_lengths = []

    with torch.no_grad():
        for history_ids, history_mask, candidate_ids, labels, cand_mask, cand_pos in tqdm(loader, desc="Evaluating JEPA"):
            history_ids = history_ids.to(device)
            history_mask = history_mask.to(device)
            candidate_ids = candidate_ids.to(device)
            labels_dev = labels.to(device)
            cand_mask = cand_mask.to(device)
            cand_pos = cand_pos.to(device)

            output = ft_model(
                history_ids=history_ids,
                history_mask=history_mask,
                candidate_ids=candidate_ids,
                cat_ids=cat_ids,
                subcat_ids=subcat_ids,
                entity_flags=entity_flags,
                labels=labels_dev,
                cand_mask=cand_mask,
                cand_pos=cand_pos,
                global_ctr=global_ctr_tensor,
            )

            scores = output["scores"].cpu().numpy()
            labels_np = labels.cpu().numpy()
            cand_mask_np = cand_mask.cpu().numpy()

            for i in range(scores.shape[0]):
                valid = cand_mask_np[i]
                if valid.sum() < 2:
                    continue
                all_impressions.append({
                    "y_true": labels_np[i][valid],
                    "y_score": scores[i][valid],
                })
                history_lengths.append(history_mask[i].sum().item())

    results = evaluate_all(all_impressions, history_lengths)
    print_results(results, "JEPA")
    return results


def evaluate_xgb(
    processed_dir: str,
    model_path: str,
) -> dict:
    """Evaluate XGBoost model on dev set."""
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("  Evaluating XGBoost model")
    print("=" * 60)

    # Load data
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    text_embeddings = data["text_embeddings"]
    cat_ids = data["article_features"]["cat_ids"]
    vocabs = data["vocabs"]
    train_users_all = data["train_users"]
    user_ids = list(train_users_all.keys())
    split_idx = int(len(user_ids) * 0.9)
    train_users = {uid: train_users_all[uid] for uid in user_ids[:split_idx]}
    dev_users = {uid: train_users_all[uid] for uid in user_ids[split_idx:]}

    from baselines.xgboost_ranker import (
        compute_user_profiles, compute_global_ctr, build_features
    )
    from data.preprocess import parse_behaviors_tsv

    num_categories = len(vocabs["cat2idx"])

    # Profiles and CTR (from training data)
    user_profiles = compute_user_profiles(
        train_users_all, cat_ids, text_embeddings, num_categories
    )

    raw_dir = processed_dir.replace("processed", "raw")
    train_behaviors = parse_behaviors_tsv(
        os.path.join(raw_dir, "mind-small", "train", "behaviors.tsv")
    )
    global_ctr = compute_global_ctr(train_behaviors, vocabs["news_id2idx"])

    # Build dev features
    X_dev, y_dev, groups_dev = build_features(
        dev_users, user_profiles, global_ctr, cat_ids,
        text_embeddings, num_categories, split="dev"
    )

    # Load model and predict
    ranker = xgb.XGBRanker()
    ranker.load_model(model_path)
    scores = ranker.predict(X_dev)

    # Reconstruct impressions from groups
    all_impressions = []
    history_lengths = []
    offset = 0
    dev_user_list = list(dev_users.values())
    user_idx = 0

    for group_size in groups_dev:
        group_scores = scores[offset:offset + group_size]
        group_labels = y_dev[offset:offset + group_size]

        if len(np.unique(group_labels)) >= 1:
            all_impressions.append({
                "y_true": group_labels,
                "y_score": group_scores,
            })
            # Approximate history length
            if user_idx < len(dev_user_list):
                hl = len(dev_user_list[user_idx].get("history_ids", []))
                history_lengths.append(hl)

        offset += group_size
        user_idx += 1

    results = evaluate_all(all_impressions, history_lengths if history_lengths else None)
    print_results(results, "XGBoost")
    return results


def compare_models(jepa_results: dict, xgb_results: dict) -> None:
    """Print side-by-side comparison."""
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(f"  {'Metric':<12} {'JEPA':>10} {'XGBoost':>10} {'Δ':>10}")
    print(f"  {'-'*42}")

    for metric in ["auc", "mrr", "ndcg@5", "ndcg@10"]:
        j = jepa_results["overall"][metric]
        x = xgb_results["overall"][metric]
        delta = j - x
        sign = "+" if delta > 0 else ""
        print(f"  {metric:<12} {j:>10.4f} {x:>10.4f} {sign}{delta:>9.4f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model", choices=["jepa", "xgb", "both"], default="both")
    parser.add_argument("--jepa-checkpoint", default="checkpoints/finetuned_model.pt")
    parser.add_argument("--xgb-model", default="checkpoints/xgb_ranker.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predictor-type", default="transformer", choices=["transformer", "mlp"])
    parser.add_argument("--scoring-mode", default="mlp", choices=["mlp", "dot"])
    args = parser.parse_args()

    jepa_results = None
    xgb_results = None

    if args.model in ["jepa", "both"]:
        jepa_results = evaluate_jepa(
            args.processed_dir, args.jepa_checkpoint, batch_size=args.batch_size,
            predictor_type=args.predictor_type, scoring_mode=args.scoring_mode
        )

    if args.model in ["xgb", "both"]:
        xgb_results = evaluate_xgb(args.processed_dir, args.xgb_model)

    if jepa_results and xgb_results:
        compare_models(jepa_results, xgb_results)
