"""Ranking head for fine-tuning on impression-level click prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingHead(nn.Module):
    def __init__(self, d_model=128, hidden_dim=256, dropout=0.1, n_extra=0, mode="mlp"):
        super().__init__()
        self.mode = mode
        self.n_extra = n_extra

        if mode == "mlp":
            # x concatenates: z_user (dim) + z_article (dim) + z_user * z_article (dim) + extra (n_extra)
            input_dim = d_model * 3 + n_extra
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        # For "dot" mode, no parameters needed other than maybe a bias or temperature

    def forward(self, z_user, z_article, extra=None):
        """
        Args:
            z_user: (batch, d_model) or (batch, num_cand, d_model)
            z_article: (batch, num_cand, d_model)
            extra: (batch, num_cand, n_extra) optional
        """
        if z_user.dim() == 2:
            num_cand = z_article.shape[1]
            z_user = z_user.unsqueeze(1).expand(-1, num_cand, -1)

        if self.mode == "dot":
            # Simple dot product between user and article embeddings
            scores = (z_user * z_article).sum(dim=-1)  # (batch, num_cand)
            return scores

        # MLP mode
        inputs = [z_user, z_article, z_user * z_article]
        if extra is not None and self.n_extra > 0:
            inputs.append(extra)
        elif self.n_extra > 0:
            # If extra is expected but not provided, pad with zeros
            batch_size, num_cand, _ = z_article.shape
            extra_zeros = torch.zeros(batch_size, num_cand, self.n_extra, device=z_article.device)
            inputs.append(extra_zeros)

        x = torch.cat(inputs, dim=-1)
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
        """Forward pass for fine-tuning."""
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
        # We pass only user and candidate representations; extra features can be added if available
        scores = self.ranking_head(z_user, z_candidates)  # (batch, num_candidates)

        # BCE loss with logits
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        return {"loss": loss, "scores": scores.detach()}
