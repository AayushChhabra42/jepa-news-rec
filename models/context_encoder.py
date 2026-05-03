"""Context encoder: 4-layer Transformer with learned positional encodings."""

import torch
import torch.nn as nn
import math


class ContextEncoder(nn.Module):
    """Transformer encoder for user click sequences.

    Processes a sequence of item embeddings with learned positional
    encodings and outputs contextualised representations. Can produce
    both per-position outputs (for the predictor) and a pooled user
    representation (for downstream tasks).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learned positional embeddings
        self.position_embed = nn.Embedding(max_seq_len, d_model)

        # Input LayerNorm (pre-norm architecture)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection (optional, for downstream)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        item_embeds: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a sequence of item embeddings.

        Args:
            item_embeds: (batch, seq_len, d_model) — item embeddings.
            padding_mask: (batch, seq_len) — bool, True for PAD positions.

        Returns:
            (batch, seq_len, d_model) — contextualised representations.
        """
        batch_size, seq_len, _ = item_embeds.shape

        # Add positional encodings
        positions = torch.arange(seq_len, device=item_embeds.device)
        pos_embeds = self.position_embed(positions)     # (seq_len, d_model)
        x = item_embeds + pos_embeds.unsqueeze(0)       # broadcast over batch

        x = self.input_norm(x)

        # Transformer expects padding_mask where True = ignore
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.output_norm(x)
        return x

    def pool(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool over non-padded positions to get a user representation.

        Args:
            x: (batch, seq_len, d_model) — encoder output.
            padding_mask: (batch, seq_len) — bool, True for PAD positions.

        Returns:
            (batch, d_model) — user-level representation.
        """
        if padding_mask is None:
            return x.mean(dim=1)

        # Invert mask: True for valid positions
        valid_mask = ~padding_mask  # (batch, seq_len)
        valid_mask = valid_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (x * valid_mask).sum(dim=1)           # (batch, d_model)
        counts = valid_mask.sum(dim=1).clamp(min=1)    # (batch, 1)
        return summed / counts
