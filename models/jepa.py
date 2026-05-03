"""JEPA model: orchestrates context encoder, predictor, and target encoder."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.context_encoder import ContextEncoder
from models.predictor import Predictor
from models.item_encoder import ItemEncoder
from utils.ema import EMAUpdater
from utils.vicreg import vicreg_regularisation


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture for sequential recommendation.

    Components:
        - Item encoder: converts news IDs → 128d embeddings
        - Context encoder: 4-layer Transformer over unmasked history
        - Predictor: 2-layer Transformer, predicts target embeddings
        - Target encoder: EMA copy of context encoder (no gradients)

    The item encoder is shared between context and target paths, but
    the target path's copy is updated via EMA along with the target
    context encoder.
    """

    def __init__(
        self,
        item_encoder: ItemEncoder,
        context_encoder_cfg: dict,
        predictor_cfg: dict,
        ema_cfg: dict,
        vicreg_lambda_var: float = 1.0,
        vicreg_lambda_cov: float = 0.04,
    ):
        super().__init__()

        self.item_encoder = item_encoder

        # Context encoder
        self.context_encoder = ContextEncoder(**context_encoder_cfg)

        # Predictor
        self.predictor = Predictor(**predictor_cfg)

        # Target encoder — deep copy, no gradient
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Target item encoder — deep copy, no gradient
        self.target_item_encoder = copy.deepcopy(self.item_encoder)
        for param in self.target_item_encoder.parameters():
            param.requires_grad_(False)

        # EMA updater
        self.ema = EMAUpdater(**ema_cfg)

        # VICReg weights
        self.vicreg_lambda_var = vicreg_lambda_var
        self.vicreg_lambda_cov = vicreg_lambda_cov

    def forward(
        self,
        ctx_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        cat_ids: torch.Tensor,
        subcat_ids: torch.Tensor,
        entity_flags: torch.Tensor,
    ) -> dict:
        """Forward pass for JEPA pretraining.

        Args:
            ctx_ids: (batch, max_seq_len) — context item indices.
            tgt_ids: (batch, max_seq_len) — target item indices.
            ctx_mask: (batch, max_seq_len) — True for valid context positions.
            tgt_mask: (batch, max_seq_len) — True for valid target positions.
            cat_ids: (num_articles,) — global category IDs.
            subcat_ids: (num_articles,) — global subcategory IDs.
            entity_flags: (num_articles,) — global entity flags.

        Returns:
            Dict with 'loss', 'pred_loss', 'vicreg_loss', and metrics.
        """
        # --- Context path ---
        ctx_embeds = self._encode_items(
            self.item_encoder, ctx_ids, cat_ids, subcat_ids, entity_flags
        )  # (batch, max_seq_len, d_model)

        ctx_padding = ~ctx_mask  # True = pad
        ctx_out = self.context_encoder(ctx_embeds, padding_mask=ctx_padding)

        # --- Target path (no gradients) ---
        with torch.no_grad():
            tgt_embeds = self._encode_items(
                self.target_item_encoder, tgt_ids, cat_ids, subcat_ids, entity_flags
            )
            tgt_padding = ~tgt_mask
            tgt_out = self.target_encoder(tgt_embeds, padding_mask=tgt_padding)

        # --- Predictor: predict target from context ---
        # Find the max number of valid targets in this batch
        num_targets = tgt_mask.sum(dim=1).max().item()
        if num_targets == 0:
            return {"loss": torch.tensor(0.0, device=ctx_ids.device)}

        predicted = self.predictor(
            context=ctx_out,
            context_padding_mask=ctx_padding,
            num_targets=num_targets,
        )  # (batch, num_targets, d_model)

        # --- Compute loss ---
        # Align predictions with targets (only valid positions)
        pred_loss = self._compute_prediction_loss(predicted, tgt_out, tgt_mask, num_targets)

        # VICReg regularisation on predictions
        # Flatten all valid predictions for the batch
        pred_flat = predicted.reshape(-1, predicted.shape[-1])
        vicreg_loss, vicreg_metrics = vicreg_regularisation(
            pred_flat,
            lambda_var=self.vicreg_lambda_var,
            lambda_cov=self.vicreg_lambda_cov,
        )

        total_loss = pred_loss + vicreg_loss

        return {
            "loss": total_loss,
            "pred_loss": pred_loss.item(),
            "vicreg_loss": vicreg_loss.item(),
            **vicreg_metrics,
        }

    def _encode_items(
        self,
        encoder: ItemEncoder,
        item_ids: torch.Tensor,
        cat_ids: torch.Tensor,
        subcat_ids: torch.Tensor,
        entity_flags: torch.Tensor,
    ) -> torch.Tensor:
        """Look up item features and encode."""
        # Gather per-item features
        cats = cat_ids[item_ids]            # (batch, seq_len)
        subcats = subcat_ids[item_ids]
        ent_flags = entity_flags[item_ids]
        return encoder(item_ids, cats, subcats, ent_flags)

    def _compute_prediction_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        tgt_mask: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        """Compute smooth L1 loss between predictions and targets.

        Only considers valid (non-padded) target positions.
        """
        # Trim target to num_targets
        target_trimmed = target[:, :num_targets, :]  # (batch, num_targets, d_model)
        mask_trimmed = tgt_mask[:, :num_targets]      # (batch, num_targets)

        # Smooth L1 per position
        loss = F.smooth_l1_loss(predicted, target_trimmed, reduction="none")  # (B, T, D)
        loss = loss.mean(dim=-1)  # (B, T) — average over embedding dims

        # Mask out invalid positions
        loss = loss * mask_trimmed.float()

        # Average over valid positions
        num_valid = mask_trimmed.float().sum().clamp(min=1)
        return loss.sum() / num_valid

    @torch.no_grad()
    def ema_update(self) -> float:
        """Update target encoder and target item encoder via EMA."""
        tau = self.ema.update(self.context_encoder, self.target_encoder)
        # Also update target item encoder
        tau_current = self.ema.tau
        for p_online, p_target in zip(
            self.item_encoder.parameters(),
            self.target_item_encoder.parameters()
        ):
            p_target.data.mul_(tau_current).add_(p_online.data, alpha=1.0 - tau_current)
        return tau

    def get_user_representation(
        self,
        item_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        cat_ids: torch.Tensor,
        subcat_ids: torch.Tensor,
        entity_flags: torch.Tensor,
    ) -> torch.Tensor:
        """Compute user representation from click history (for inference).

        Args:
            item_ids: (batch, seq_len) — click history item indices.
            padding_mask: (batch, seq_len) — True for valid positions.
            cat_ids: (num_articles,) — global category IDs.
            subcat_ids: (num_articles,) — global subcategory IDs.
            entity_flags: (num_articles,) — global entity flags.

        Returns:
            (batch, d_model) — user embeddings.
        """
        embeds = self._encode_items(
            self.item_encoder, item_ids, cat_ids, subcat_ids, entity_flags
        )
        pad = ~padding_mask
        ctx_out = self.context_encoder(embeds, padding_mask=pad)
        return self.context_encoder.pool(ctx_out, padding_mask=pad)
