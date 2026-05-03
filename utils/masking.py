"""Temporal masking for JEPA pretraining on click sequences."""

import torch
import math


def temporal_mask(seq_len: int, min_ratio: float = 0.2, max_ratio: float = 0.4) -> tuple[int, int]:
    """Compute a temporal mask split point.

    Masks the last `ratio` fraction of the sequence, where
    ratio ~ Uniform(min_ratio, max_ratio).

    Args:
        seq_len: Length of the (unpadded) click history.
        min_ratio: Minimum fraction of the sequence to mask.
        max_ratio: Maximum fraction of the sequence to mask.

    Returns:
        (ctx_end, mask_start) — indices into the sequence.
        Context = seq[:ctx_end], Target = seq[mask_start:]
        (ctx_end == mask_start in temporal masking)
    """
    ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
    mask_len = max(1, math.ceil(seq_len * ratio))
    split = seq_len - mask_len
    return split, split


def apply_temporal_mask(
    item_ids: torch.Tensor,
    lengths: torch.Tensor,
    min_ratio: float = 0.2,
    max_ratio: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply temporal masking to a batch of padded sequences.

    Args:
        item_ids: (batch, max_seq_len) — padded item ID sequences.
        lengths: (batch,) — actual lengths of each sequence.
        min_ratio: Minimum mask fraction.
        max_ratio: Maximum mask fraction.

    Returns:
        ctx_ids: (batch, max_seq_len) — context items (masked positions zeroed).
        tgt_ids: (batch, max_seq_len) — target items (context positions zeroed).
        ctx_mask: (batch, max_seq_len) — bool, True for valid context positions.
        tgt_mask: (batch, max_seq_len) — bool, True for valid target positions.
    """
    batch_size, max_len = item_ids.shape
    ctx_ids = torch.zeros_like(item_ids)
    tgt_ids = torch.zeros_like(item_ids)
    ctx_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=item_ids.device)
    tgt_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=item_ids.device)

    for i in range(batch_size):
        L = lengths[i].item()
        if L < 2:
            # Degenerate case — put everything in context
            ctx_ids[i, :L] = item_ids[i, :L]
            ctx_mask[i, :L] = True
            continue

        split, _ = temporal_mask(L, min_ratio, max_ratio)
        split = max(1, split)  # At least 1 context item

        ctx_ids[i, :split] = item_ids[i, :split]
        ctx_mask[i, :split] = True

        tgt_len = L - split
        tgt_ids[i, :tgt_len] = item_ids[i, split:L]
        tgt_mask[i, :tgt_len] = True

    return ctx_ids, tgt_ids, ctx_mask, tgt_mask
