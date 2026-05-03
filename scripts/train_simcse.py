"""Stage 2: SimCSE co-click pretraining for item embeddings."""

import os
import sys
import pickle
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.item_encoder import ItemEncoder
from models.simcse import SimCSEModel

logger = logging.getLogger(__name__)


class CoClickDataset(Dataset):
    """Dataset of co-click pairs for SimCSE training."""

    def __init__(self, pairs, cat_ids, subcat_ids, entity_flags, hard_negatives=None, num_hard_neg=5):
        self.pairs = pairs
        self.cat_ids = cat_ids
        self.subcat_ids = subcat_ids
        self.entity_flags = entity_flags
        self.hard_negatives = hard_negatives
        self.num_hard_neg = num_hard_neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_idx, p_idx = self.pairs[idx]

        item = {
            "anchor_id": a_idx,
            "positive_id": p_idx,
            "anchor_cat": self.cat_ids[a_idx],
            "positive_cat": self.cat_ids[p_idx],
            "anchor_subcat": self.subcat_ids[a_idx],
            "positive_subcat": self.subcat_ids[p_idx],
            "anchor_ent": self.entity_flags[a_idx],
            "positive_ent": self.entity_flags[p_idx],
        }

        if self.hard_negatives is not None and a_idx in self.hard_negatives:
            negs = self.hard_negatives[a_idx]
            # Sample num_hard_neg negatives
            if len(negs) >= self.num_hard_neg:
                chosen = np.random.choice(negs, self.num_hard_neg, replace=False)
            else:
                chosen = np.random.choice(negs, self.num_hard_neg, replace=True)
            item["hard_neg_ids"] = chosen
            item["hard_neg_cats"] = self.cat_ids[chosen]
            item["hard_neg_subcats"] = self.subcat_ids[chosen]
            item["hard_neg_ents"] = self.entity_flags[chosen]

        return item


def collate_fn(batch):
    """Collate co-click samples."""
    result = {
        "anchor_ids": torch.tensor([b["anchor_id"] for b in batch], dtype=torch.long),
        "positive_ids": torch.tensor([b["positive_id"] for b in batch], dtype=torch.long),
        "anchor_cats": torch.tensor([b["anchor_cat"] for b in batch], dtype=torch.long),
        "positive_cats": torch.tensor([b["positive_cat"] for b in batch], dtype=torch.long),
        "anchor_subcats": torch.tensor([b["anchor_subcat"] for b in batch], dtype=torch.long),
        "positive_subcats": torch.tensor([b["positive_subcat"] for b in batch], dtype=torch.long),
        "anchor_ent_flags": torch.tensor([b["anchor_ent"] for b in batch], dtype=torch.float),
        "positive_ent_flags": torch.tensor([b["positive_ent"] for b in batch], dtype=torch.float),
    }

    if "hard_neg_ids" in batch[0]:
        result["hard_negative_ids"] = torch.tensor(
            np.stack([b["hard_neg_ids"] for b in batch]), dtype=torch.long
        )
        result["hard_negative_cats"] = torch.tensor(
            np.stack([b["hard_neg_cats"] for b in batch]), dtype=torch.long
        )
        result["hard_negative_subcats"] = torch.tensor(
            np.stack([b["hard_neg_subcats"] for b in batch]), dtype=torch.long
        )
        result["hard_negative_ent_flags"] = torch.tensor(
            np.stack([b["hard_neg_ents"] for b in batch]), dtype=torch.float
        )

    return result


def train_simcse(
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 256,
    temperature: float = 0.05,
    use_hard_negatives: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train SimCSE on co-click pairs."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    text_embeddings = data["text_embeddings"]
    cat_ids = data["article_features"]["cat_ids"]
    subcat_ids = data["article_features"]["subcat_ids"]
    entity_flags = data["article_features"]["entity_flags"]
    coclick_pairs = data["coclick_pairs"]
    hard_negatives = data["hard_negatives"] if use_hard_negatives else None
    vocabs = data["vocabs"]

    print(f"  Articles: {text_embeddings.shape[0]}")
    print(f"  Co-click pairs: {len(coclick_pairs)}")
    print(f"  Categories: {len(vocabs['cat2idx'])}")
    print(f"  Subcategories: {len(vocabs['subcat2idx'])}")

    # Build model
    item_encoder = ItemEncoder(
        text_embeddings=text_embeddings,
        num_categories=len(vocabs["cat2idx"]),
        num_subcategories=len(vocabs["subcat2idx"]),
    )
    model = SimCSEModel(item_encoder, temperature=temperature).to(device)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset and loader
    dataset = CoClickDataset(
        coclick_pairs, cat_ids, subcat_ids, entity_flags,
        hard_negatives=hard_negatives,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    # Training loop
    print(f"\nStarting SimCSE training ({epochs} epochs, {len(loader)} batches/epoch)")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Move to device
            kwargs = {
                "anchor_ids": batch["anchor_ids"].to(device),
                "positive_ids": batch["positive_ids"].to(device),
                "anchor_cats": batch["anchor_cats"].to(device),
                "positive_cats": batch["positive_cats"].to(device),
                "anchor_subcats": batch["anchor_subcats"].to(device),
                "positive_subcats": batch["positive_subcats"].to(device),
                "anchor_ent_flags": batch["anchor_ent_flags"].to(device),
                "positive_ent_flags": batch["positive_ent_flags"].to(device),
            }
            if "hard_negative_ids" in batch:
                kwargs["hard_negative_ids"] = batch["hard_negative_ids"].to(device)
                kwargs["hard_negative_cats"] = batch["hard_negative_cats"].to(device)
                kwargs["hard_negative_subcats"] = batch["hard_negative_subcats"].to(device)
                kwargs["hard_negative_ent_flags"] = batch["hard_negative_ent_flags"].to(device)

            output = model(**kwargs)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += output["accuracy"]
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{output['accuracy']:.3f}")

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={avg_acc:.3f}")

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "simcse_item_encoder.pt")
    torch.save({
        "item_encoder_state": model.item_encoder.state_dict(),
        "epoch": epochs,
        "loss": avg_loss,
    }, ckpt_path)
    print(f"\n✓ SimCSE training complete. Saved to {ckpt_path}")

    return model.item_encoder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train SimCSE on co-click pairs")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--no-hard-negatives", action="store_true")
    args = parser.parse_args()

    train_simcse(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        temperature=args.temperature,
        use_hard_negatives=not args.no_hard_negatives,
    )
