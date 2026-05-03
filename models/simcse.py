"""SimCSE co-click pretraining for item embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSEModel(nn.Module):
    """Contrastive learning model for item embeddings using co-click signal.

    Trains item embeddings so that co-clicked articles (same user, same
    session) are closer in the embedding space, while random articles
    are pushed apart.

    Uses InfoNCE loss with in-batch negatives and optional hard negatives.
    """

    def __init__(
        self,
        item_encoder: nn.Module,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.item_encoder = item_encoder
        self.temperature = temperature

    def forward(
        self,
        anchor_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        anchor_cats: torch.Tensor,
        positive_cats: torch.Tensor,
        anchor_subcats: torch.Tensor,
        positive_subcats: torch.Tensor,
        anchor_ent_flags: torch.Tensor,
        positive_ent_flags: torch.Tensor,
        hard_negative_ids: torch.Tensor | None = None,
        hard_negative_cats: torch.Tensor | None = None,
        hard_negative_subcats: torch.Tensor | None = None,
        hard_negative_ent_flags: torch.Tensor | None = None,
    ) -> dict:
        """Compute InfoNCE contrastive loss.

        Args:
            anchor_ids: (batch,) — anchor article indices.
            positive_ids: (batch,) — co-clicked positive article indices.
            anchor_cats, positive_cats: (batch,) — category indices.
            anchor_subcats, positive_subcats: (batch,) — subcategory indices.
            anchor_ent_flags, positive_ent_flags: (batch,) — entity flags.
            hard_negative_ids: (batch, num_hard_neg) — optional hard negs.
            hard_negative_cats, etc: corresponding features for hard negs.

        Returns:
            Dict with 'loss' and 'accuracy'.
        """
        # Encode anchors and positives
        z_anchor = self.item_encoder(
            anchor_ids, anchor_cats, anchor_subcats, anchor_ent_flags
        )  # (batch, dim)
        z_positive = self.item_encoder(
            positive_ids, positive_cats, positive_subcats, positive_ent_flags
        )  # (batch, dim)

        # Normalise for cosine similarity
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)

        # In-batch negatives: similarity matrix
        # (batch, batch) — each row i, col j = sim(anchor_i, positive_j)
        sim_matrix = torch.matmul(z_anchor, z_positive.T) / self.temperature

        # Add hard negatives if provided
        if hard_negative_ids is not None:
            z_hard_neg = self.item_encoder(
                hard_negative_ids, hard_negative_cats,
                hard_negative_subcats, hard_negative_ent_flags
            )  # (batch, num_hard_neg, dim)
            z_hard_neg = F.normalize(z_hard_neg, dim=-1)

            # (batch, num_hard_neg)
            hard_neg_sim = torch.bmm(
                z_anchor.unsqueeze(1),  # (batch, 1, dim)
                z_hard_neg.transpose(1, 2)  # (batch, dim, num_hard_neg)
            ).squeeze(1) / self.temperature

            # Concat: (batch, batch + num_hard_neg)
            sim_matrix = torch.cat([sim_matrix, hard_neg_sim], dim=1)

        # Labels: diagonal is positive (index i)
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)

        # Accuracy (for monitoring)
        with torch.no_grad():
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}
