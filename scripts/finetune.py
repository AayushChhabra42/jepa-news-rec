"""Stage 4: Fine-tune ranking head on impression-level click prediction."""

import os
import sys
import pickle
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.item_encoder import ItemEncoder
from models.jepa import JEPA
from models.ranking_head import RankingHead, FineTuneModel
from baselines.xgboost_ranker import compute_global_ctr
from data.preprocess import parse_behaviors_tsv

logger = logging.getLogger(__name__)


class ImpressionDataset(Dataset):
    """Dataset of impression-level samples for fine-tuning.

    Each sample has: user history + candidate articles + labels.
    """

    def __init__(
        self,
        user_sequences: dict,
        max_seq_len: int = 50,
        neg_sample_ratio: int = 4,
    ):
        self.samples = []
        self.max_seq_len = max_seq_len
        self.neg_sample_ratio = neg_sample_ratio

        for uid, data in user_sequences.items():
            for impression in data["impressions"]:
                hist = impression["history_ids"][-max_seq_len:]
                candidates = impression["candidates"]
                labels = impression["labels"]

                if not hist or not candidates:
                    continue
                if sum(labels) == 0:
                    continue  # No positive

                self.samples.append({
                    "history": hist,
                    "candidates": candidates,
                    "labels": labels,
                })

        print(f"  ImpressionDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Subsample negatives to balance
        candidates = sample["candidates"]
        labels = sample["labels"]

        positives = [(pos, c, l) for pos, (c, l) in enumerate(zip(candidates, labels)) if l == 1]
        negatives = [(pos, c, l) for pos, (c, l) in enumerate(zip(candidates, labels)) if l == 0]

        # Keep all positives, subsample negatives
        max_neg = len(positives) * self.neg_sample_ratio
        if len(negatives) > max_neg:
            neg_indices = np.random.choice(len(negatives), max_neg, replace=False)
            negatives = [negatives[i] for i in neg_indices]

        selected = positives + negatives
        np.random.shuffle(selected)

        return {
            "history": sample["history"],
            "candidates": [c for pos, c, l in selected],
            "labels": [l for pos, c, l in selected],
            "positions": [pos for pos, c, l in selected],
        }


def collate_impressions(batch, max_seq_len=50):
    """Collate impression samples with padding."""
    batch_size = len(batch)
    max_cand = max(len(b["candidates"]) for b in batch)

    history_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    history_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    candidate_ids = torch.zeros(batch_size, max_cand, dtype=torch.long)
    labels = torch.zeros(batch_size, max_cand, dtype=torch.float)
    cand_mask = torch.zeros(batch_size, max_cand, dtype=torch.bool)
    cand_pos = torch.zeros(batch_size, max_cand, dtype=torch.long)

    for i, sample in enumerate(batch):
        h = sample["history"]
        hlen = min(len(h), max_seq_len)
        history_ids[i, :hlen] = torch.tensor(h[:hlen])
        history_mask[i, :hlen] = True

        c = sample["candidates"]
        l = sample["labels"]
        p = sample["positions"]
        clen = len(c)
        candidate_ids[i, :clen] = torch.tensor(c)
        labels[i, :clen] = torch.tensor(l, dtype=torch.float)
        cand_mask[i, :clen] = True
        cand_pos[i, :clen] = torch.tensor(p, dtype=torch.long)

    return history_ids, history_mask, candidate_ids, labels, cand_mask, cand_pos


def finetune(
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    jepa_checkpoint: str = None,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 128,
    max_seq_len: int = 50,
    neg_sample_ratio: int = 4,
    scoring_mode: str = "mlp",
    predictor_type: str = "transformer",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Fine-tune ranking head on labeled impressions."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data
    print("Loading preprocessed data...")
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    text_embeddings = data["text_embeddings"]
    cat_ids_np = data["article_features"]["cat_ids"]
    subcat_ids_np = data["article_features"]["subcat_ids"]
    entity_flags_np = data["article_features"]["entity_flags"]
    vocabs = data["vocabs"]
    train_users = data["train_users"]

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

    # Build JEPA model and load weights
    print("Building model...")
    item_encoder = ItemEncoder(
        text_embeddings=text_embeddings,
        num_categories=len(vocabs["cat2idx"]),
        num_subcategories=len(vocabs["subcat2idx"]),
    )

    jepa_model = JEPA(
        item_encoder=item_encoder,
        context_encoder_cfg={
            "d_model": 128, "nhead": 4, "num_layers": 4,
            "d_ff": 512, "dropout": 0.1, "max_seq_len": max_seq_len,
        },
        predictor_cfg={
            "type": predictor_type,
            "d_model": 128, "nhead": 4, "num_layers": 2,
            "d_ff": 256, "dropout": 0.1, "max_target_len": max_seq_len,
        },
        ema_cfg={"tau_start": 0.996, "tau_end": 0.9999, "total_steps": 1},
    )

    if jepa_checkpoint and os.path.exists(jepa_checkpoint):
        print(f"  Loading JEPA from {jepa_checkpoint}")
        ckpt = torch.load(jepa_checkpoint, map_location="cpu")
        jepa_model.load_state_dict(ckpt["model_state"])

    # Build ranking head
    ranking_head = RankingHead(d_model=128, hidden_dim=256, dropout=0.1, mode=scoring_mode)

    # Fine-tune model
    ft_model = FineTuneModel(jepa_model, ranking_head).to(device)

    trainable_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    print(f"  Trainable params (ranking head only): {trainable_params:,}")

    # Dataset
    dataset = ImpressionDataset(train_users, max_seq_len=max_seq_len, neg_sample_ratio=neg_sample_ratio)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: collate_impressions(b, max_seq_len),
    )

    # Optimizer (only ranking head params)
    optimizer = AdamW(ft_model.ranking_head.parameters(), lr=lr, weight_decay=0.01)

    # Training
    print(f"\nStarting fine-tuning ({epochs} epochs)")
    print("=" * 60)

    for epoch in range(epochs):
        # 2-stage unfreezing
        if epoch == 2:
            print("\n  [Unfreezing JEPA encoder for differential finetuning]")
            for param in ft_model.jepa.parameters():
                param.requires_grad_(True)
            optimizer = AdamW([
                {"params": ft_model.jepa.item_encoder.parameters(), "lr": lr * 0.01},
                {"params": ft_model.jepa.context_encoder.parameters(), "lr": lr * 0.1},
                {"params": ft_model.ranking_head.parameters(), "lr": lr}
            ], weight_decay=0.01)

        ft_model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for history_ids, history_mask, candidate_ids, labels, cand_mask, cand_pos in pbar:
            history_ids = history_ids.to(device)
            history_mask = history_mask.to(device)
            candidate_ids = candidate_ids.to(device)
            labels = labels.to(device)
            cand_mask = cand_mask.to(device)
            cand_pos = cand_pos.to(device)

            output = ft_model(
                history_ids=history_ids,
                history_mask=history_mask,
                candidate_ids=candidate_ids,
                cat_ids=cat_ids,
                subcat_ids=subcat_ids,
                entity_flags=entity_flags,
                labels=labels,
                cand_mask=cand_mask,
                cand_pos=cand_pos,
                global_ctr=global_ctr_tensor,
            )

            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_model.ranking_head.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"  Epoch {epoch+1}: loss={epoch_loss / num_batches:.4f}")

    # Save
    ckpt_path = os.path.join(checkpoint_dir, "finetuned_model.pt")
    torch.save({
        "jepa_state": ft_model.jepa.state_dict(),
        "ranking_head_state": ft_model.ranking_head.state_dict(),
        "epoch": epochs,
    }, ckpt_path)
    print(f"\n✓ Fine-tuning complete. Saved to {ckpt_path}")

    return ft_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fine-tune ranking head")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--jepa-checkpoint", default="checkpoints/jepa_best.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scoring-mode", default="mlp", choices=["mlp", "dot"])
    parser.add_argument("--predictor-type", default="transformer", choices=["transformer", "mlp"])
    args = parser.parse_args()

    finetune(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        jepa_checkpoint=args.jepa_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        scoring_mode=args.scoring_mode,
        predictor_type=args.predictor_type,
    )
