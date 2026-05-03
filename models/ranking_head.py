"""Ranking head for fine-tuning on impression-level click prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingHead(nn.Module):
    """MLP ranking head for click prediction.

    Takes a user representation and a candidate article embedding,
    and outputs a click probability score.

    Supports two modes:
      - "mlp": concat(z_user, z_article) → MLP → score
      - "dot": dot(z_user, z_article) → score
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        mode: str = "mlp",
    ):
        super().__init__()
        self.mode = mode

        if mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        # dot mode needs no parameters

    def forward(
        self,
        z_user: torch.Tensor,
        z_article: torch.Tensor,
    ) -> torch.Tensor:
        """Compute click score.

        Args:
            z_user: (batch, d_model) or (batch, 1, d_model).
            z_article: (batch, d_model) or (batch, num_candidates, d_model).

        Returns:
            (batch,) or (batch, num_candidates) — click scores.
        """
        if self.mode == "dot":
            if z_user.dim() == 2 and z_article.dim() == 3:
                # (batch, 1, d) × (batch, d, num_cand) → (batch, num_cand)
                return torch.bmm(
                    z_user.unsqueeze(1), z_article.transpose(1, 2)
                ).squeeze(1)
            return (z_user * z_article).sum(dim=-1)

        # MLP mode
        if z_user.dim() == 2 and z_article.dim() == 3:
            # Expand z_user to match candidates
            num_cand = z_article.shape[1]
            z_user_exp = z_user.unsqueeze(1).expand(-1, num_cand, -1)
            combined = torch.cat([z_user_exp, z_article], dim=-1)
            scores = self.mlp(combined).squeeze(-1)  # (batch, num_cand)
            return scores

        combined = torch.cat([z_user, z_article], dim=-1)
        return self.mlp(combined).squeeze(-1)


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
