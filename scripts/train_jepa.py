"""Stage 3: JEPA sequential pretraining."""

import os
import sys
import pickle
import argparse
import logging
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.item_encoder import ItemEncoder
from models.jepa import JEPA
from utils.masking import apply_temporal_mask
from utils.collapse_monitor import CollapseMonitor

logger = logging.getLogger(__name__)


class UserSequenceDataset(Dataset):
    """Dataset of user click histories for JEPA pretraining."""

    def __init__(self, user_sequences: dict, max_seq_len: int = 50, min_history: int = 5):
        self.sequences = []
        for uid, data in user_sequences.items():
            hist = data["history_ids"]
            if len(hist) >= min_history:
                # Truncate and pad
                hist = hist[-max_seq_len:]
                self.sequences.append({
                    "user_id": uid,
                    "history": hist,
                    "length": len(hist),
                })
        print(f"  UserSequenceDataset: {len(self.sequences)} users (min history={min_history})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_sequences(batch, max_seq_len=50):
    """Pad user sequences to max_seq_len."""
    item_ids = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, sample in enumerate(batch):
        L = sample["length"]
        item_ids[i, :L] = torch.tensor(sample["history"], dtype=torch.long)
        lengths[i] = L

    return item_ids, lengths


def train_jepa(
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    simcse_checkpoint: str = None,
    epochs: int = 10,
    lr: float = 1e-4,
    item_embed_lr_scale: float = 0.1,
    batch_size: int = 256,
    max_seq_len: int = 50,
    min_mask_ratio: float = 0.2,
    max_mask_ratio: float = 0.4,
    ema_tau_start: float = 0.996,
    ema_tau_end: float = 0.9999,
    vicreg_lambda_var: float = 1.0,
    vicreg_lambda_cov: float = 0.04,
    collapse_threshold: float = 0.05,
    collapse_check_interval: int = 500,
    warmup_steps: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train JEPA on user click sequences."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Load data ---
    print("Loading preprocessed data...")
    with open(os.path.join(processed_dir, "processed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    text_embeddings = data["text_embeddings"]
    cat_ids_np = data["article_features"]["cat_ids"]
    subcat_ids_np = data["article_features"]["subcat_ids"]
    entity_flags_np = data["article_features"]["entity_flags"]
    vocabs = data["vocabs"]
    train_users = data["train_users"]

    # Move article-level features to device (used as lookup tables)
    cat_ids = torch.tensor(cat_ids_np, dtype=torch.long, device=device)
    subcat_ids = torch.tensor(subcat_ids_np, dtype=torch.long, device=device)
    entity_flags = torch.tensor(entity_flags_np, dtype=torch.float, device=device)

    # --- Build model ---
    print("Building JEPA model...")
    item_encoder = ItemEncoder(
        text_embeddings=text_embeddings,
        num_categories=len(vocabs["cat2idx"]),
        num_subcategories=len(vocabs["subcat2idx"]),
    )

    # Load SimCSE pretrained weights if available
    if simcse_checkpoint and os.path.exists(simcse_checkpoint):
        print(f"  Loading SimCSE weights from {simcse_checkpoint}")
        ckpt = torch.load(simcse_checkpoint, map_location="cpu")
        item_encoder.load_state_dict(ckpt["item_encoder_state"])

    # Estimate total steps for EMA schedule
    dataset = UserSequenceDataset(train_users, max_seq_len=max_seq_len)
    total_steps = epochs * (len(dataset) // batch_size + 1)

    model = JEPA(
        item_encoder=item_encoder,
        context_encoder_cfg={
            "d_model": 128, "nhead": 4, "num_layers": 4,
            "d_ff": 512, "dropout": 0.1, "max_seq_len": max_seq_len,
        },
        predictor_cfg={
            "d_model": 128, "nhead": 4, "num_layers": 2,
            "d_ff": 256, "dropout": 0.1, "max_target_len": max_seq_len,
        },
        ema_cfg={
            "tau_start": ema_tau_start,
            "tau_end": ema_tau_end,
            "total_steps": total_steps,
        },
        vicreg_lambda_var=vicreg_lambda_var,
        vicreg_lambda_cov=vicreg_lambda_cov,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {num_params:,}")
    print(f"  Total steps: {total_steps}")

    # --- Optimizer with differential LR ---
    # Item encoder gets lower LR
    param_groups = [
        {"params": model.item_encoder.parameters(), "lr": lr * item_embed_lr_scale},
        {"params": model.context_encoder.parameters(), "lr": lr},
        {"params": model.predictor.parameters(), "lr": lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=0.01)

    # Scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Data loader ---
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=lambda b: collate_sequences(b, max_seq_len),
    )

    # --- Collapse monitor ---
    collapse_monitor = CollapseMonitor(threshold=collapse_threshold)

    # Fixed eval batch for collapse monitoring
    eval_item_ids, eval_lengths = next(iter(
        DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True,
                   collate_fn=lambda b: collate_sequences(b, max_seq_len))
    ))
    eval_item_ids = eval_item_ids.to(device)
    eval_lengths = eval_lengths.to(device)

    # --- Training loop ---
    print(f"\nStarting JEPA training ({epochs} epochs)")
    print("=" * 60)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_pred_loss = 0
        epoch_vicreg_loss = 0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for item_ids, lengths in pbar:
            item_ids = item_ids.to(device)
            lengths = lengths.to(device)

            # Apply temporal masking
            ctx_ids, tgt_ids, ctx_mask, tgt_mask = apply_temporal_mask(
                item_ids, lengths, min_mask_ratio, max_mask_ratio
            )

            # Forward
            output = model(
                ctx_ids=ctx_ids,
                tgt_ids=tgt_ids,
                ctx_mask=ctx_mask,
                tgt_mask=tgt_mask,
                cat_ids=cat_ids,
                subcat_ids=subcat_ids,
                entity_flags=entity_flags,
            )

            loss = output["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update
            tau = model.ema_update()

            # Logging
            epoch_loss += loss.item()
            epoch_pred_loss += output.get("pred_loss", 0)
            epoch_vicreg_loss += output.get("vicreg_loss", 0)
            num_batches += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                tau=f"{tau:.5f}",
            )

            # Collapse check
            if global_step % collapse_check_interval == 0:
                model.eval()
                with torch.no_grad():
                    eval_mask = torch.zeros_like(eval_item_ids, dtype=torch.bool)
                    for i in range(eval_item_ids.shape[0]):
                        eval_mask[i, :eval_lengths[i]] = True
                    z_users = model.get_user_representation(
                        eval_item_ids, eval_mask, cat_ids, subcat_ids, entity_flags
                    )
                metrics = collapse_monitor.check(z_users, global_step)
                if metrics["collapsed"]:
                    print(f"\n⚠️  Collapse detected! Consider reducing EMA tau or predictor capacity.")
                model.train()

        # Epoch summary
        elapsed = time.time() - t0
        avg_loss = epoch_loss / num_batches
        print(
            f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"pred_loss={epoch_pred_loss/num_batches:.4f}, "
            f"vicreg_loss={epoch_vicreg_loss/num_batches:.4f}, "
            f"time={elapsed:.1f}s"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "jepa_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "global_step": global_step,
            }, ckpt_path)
            print(f"  Saved best model to {ckpt_path}")

    # Save final
    final_path = os.path.join(checkpoint_dir, "jepa_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epochs,
        "loss": avg_loss,
        "global_step": global_step,
        "collapse_history": collapse_monitor.history,
    }, final_path)
    print(f"\n✓ JEPA training complete. Final model saved to {final_path}")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train JEPA pretraining")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--simcse-checkpoint", default="checkpoints/simcse_item_encoder.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=50)
    args = parser.parse_args()

    train_jepa(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        simcse_checkpoint=args.simcse_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
