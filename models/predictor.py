"""Predictor: Transformer or MLP based prediction of target embeddings."""

import torch
import torch.nn as nn


class TransformerPredictor(nn.Module):
    """JEPA predictor network using a Transformer decoder.

    Takes the context encoder output and a set of learnable positional
    queries (one per masked target position), and predicts the target
    encoder output at those positions.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,        # Narrower than context encoder
        dropout: float = 0.1,
        max_target_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable position queries for target positions
        self.target_position_embed = nn.Embedding(max_target_len, d_model)

        # Cross-attention: queries attend to context encoder output
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
        num_targets: int,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        device = context.device

        # Create positional queries for target positions
        target_positions = torch.arange(num_targets, device=device)
        queries = self.target_position_embed(target_positions)  # (num_targets, d_model)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_targets, d_model)

        # Cross-attend from target queries to context
        predicted = self.decoder(
            tgt=queries,
            memory=context,
            memory_key_padding_mask=context_padding_mask,
        )

        predicted = self.output_norm(predicted)
        predicted = self.output_proj(predicted)
        return predicted


class MLPPredictor(nn.Module):
    """JEPA predictor network using an MLP.

    Uses learnable positional queries concatenated with a pooled context
    representation to predict target embeddings.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_target_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable position queries for target positions
        self.target_position_embed = nn.Embedding(max_target_len, d_model)

        # MLP: takes [query_embed; pooled_context]
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.LayerNorm(d_ff),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
        num_targets: int,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        device = context.device

        # 1. Pool context (mean pooling over valid tokens)
        if context_padding_mask is not None:
            valid_mask = (~context_padding_mask).unsqueeze(-1).float()  # (B, L, 1)
            summed = (context * valid_mask).sum(dim=1)  # (B, D)
            counts = valid_mask.sum(dim=1).clamp(min=1)  # (B, 1)
            pooled = summed / counts
        else:
            pooled = context.mean(dim=1)  # (B, D)

        # 2. Prepare queries
        target_positions = torch.arange(num_targets, device=device)
        queries = self.target_position_embed(target_positions)  # (T, D)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, D)

        # 3. Concatenate pooled context with each query
        # Expand pooled context to (B, T, D)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, num_targets, -1)
        x = torch.cat([queries, pooled_expanded], dim=-1)  # (B, T, 2*D)

        # 4. Pass through MLP
        predicted = self.net(x)  # (B, T, D)
        return predicted


class Predictor(nn.Module):
    """Factory wrapper for JEPA predictors."""

    def __init__(self, type: str = "transformer", **kwargs):
        super().__init__()
        # Pop nhead and num_layers if they aren't used by MLP to avoid unexpected kwarg errors
        if type == "mlp":
            mlp_kwargs = {k: v for k, v in kwargs.items() if k in ["d_model", "d_ff", "dropout", "max_target_len"]}
            self.model = MLPPredictor(**mlp_kwargs)
        else:
            self.model = TransformerPredictor(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
