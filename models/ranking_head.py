"""Ranking head for fine-tuning on impression-level click prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingHead(nn.Module):
    def __init__(self, embed_dim=128, n_extra=3):
        super().__init__()
        self.mlp = nn.Sequential(
            # Fixed from embed_dim * 2 to embed_dim * 3 because x concatenates:
            # z_user (dim) + z_article (dim) + z_user * z_article (dim) + extra (n_extra)
            nn.Linear(embed_dim * 3 + n_extra, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, z_user, z_article, extra):
        # extra: [position, global_ctr, category_match]
        
        # We need to make sure z_user matches z_article shape if needed, like the original
        if z_user.dim() == 2 and z_article.dim() == 3:
            num_cand = z_article.shape[1]
            z_user = z_user.unsqueeze(1).expand(-1, num_cand, -1)
            
        x = torch.cat([z_user, z_article, z_user * z_article, extra], dim=-1)
        return self.mlp(x).squeeze(-1)


class FineTuneModel(nn.Module):
    """Wraps frozen JEPA context encoder + trainable ranking head.

    During fine-tuning, the context encoder is frozen and only the
    ranking head is trained on impression-level click/non-click labels.
    """

    def __init__(
        self,
        jepa_model,
        ranking_head: RankingHead,
    ):
        super().__init__()
        self.jepa = jepa_model
        self.ranking_head = ranking_head

        # Freeze JEPA components
        for param in self.jepa.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        cat_ids: torch.Tensor,
        subcat_ids: torch.Tensor,
        entity_flags: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Forward pass for fine-tuning.

        Args:
            history_ids: (batch, seq_len) — user click history.
            history_mask: (batch, seq_len) — True for valid history positions.
            candidate_ids: (batch, num_candidates) — candidate article indices.
            cat_ids: (num_articles,) — global category IDs.
            subcat_ids: (num_articles,) — global subcategory IDs.
            entity_flags: (num_articles,) — global entity flags.
            labels: (batch, num_candidates) — 1=clicked, 0=not clicked.

        Returns:
            Dict with 'loss', 'scores'.
        """
        # Get user representation (frozen)
        with torch.no_grad():
            z_user = self.jepa.get_user_representation(
                history_ids, history_mask, cat_ids, subcat_ids, entity_flags
            )  # (batch, d_model)

            # Get candidate embeddings (frozen)
            cand_cats = cat_ids[candidate_ids]
            cand_subcats = subcat_ids[candidate_ids]
            cand_ent = entity_flags[candidate_ids]
            z_candidates = self.jepa.item_encoder(
                candidate_ids, cand_cats, cand_subcats, cand_ent
            )  # (batch, num_candidates, d_model)

        # Score with ranking head (trainable)
        scores = self.ranking_head(z_user, z_candidates)  # (batch, num_candidates)

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        return {"loss": loss, "scores": scores.detach()}
