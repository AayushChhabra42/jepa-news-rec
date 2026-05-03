"""Predictor: 2-layer Transformer with positional queries for per-target prediction."""

import torch
import torch.nn as nn


# class Predictor(nn.Module):
#     """JEPA predictor network.
# 
#     Takes the context encoder output and a set of learnable positional
#     queries (one per masked target position), and predicts the target
#     encoder output at those positions.
# 
#     Key design choice: this is a *Transformer* (not an MLP), following
#     the I-JEPA architecture. The FF dimension is intentionally narrower
#     than the context encoder (256 vs 512) to create a bottleneck that
#     helps prevent collapse.
#     """
# 
#     def __init__(
#         self,
#         d_model: int = 128,
#         nhead: int = 4,
#         num_layers: int = 2,
#         d_ff: int = 256,        # Narrower than context encoder
#         dropout: float = 0.1,
#         max_target_len: int = 50,
#     ):
#         super().__init__()
#         self.d_model = d_model
# 
#         # Learnable position queries for target positions
#         self.target_position_embed = nn.Embedding(max_target_len, d_model)
# 
#         # Cross-attention: queries attend to context encoder output
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_ff,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )
#         self.decoder = nn.TransformerDecoder(
#             decoder_layer, num_layers=num_layers
#         )
# 
#         self.output_norm = nn.LayerNorm(d_model)
#         self.output_proj = nn.Linear(d_model, d_model)
# 
#     def forward(
#         self,
#         context: torch.Tensor,
#         context_padding_mask: torch.Tensor | None,
#         num_targets: int,
#     ) -> torch.Tensor:
#         """Predict target embeddings from context.
# 
#         Args:
#             context: (batch, ctx_len, d_model) — context encoder output.
#             context_padding_mask: (batch, ctx_len) — True for PAD positions.
#             num_targets: Number of target positions to predict.
# 
#         Returns:
#             (batch, num_targets, d_model) — predicted target embeddings.
#         """
#         batch_size = context.shape[0]
#         device = context.device
# 
#         # Create positional queries for target positions
#         target_positions = torch.arange(num_targets, device=device)
#         queries = self.target_position_embed(target_positions)  # (num_targets, d_model)
#         queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_targets, d_model)
# 
#         # Cross-attend from target queries to context
#         predicted = self.decoder(
#             tgt=queries,
#             memory=context,
#             memory_key_padding_mask=context_padding_mask,
#         )
# 
#         predicted = self.output_norm(predicted)
#         predicted = self.output_proj(predicted)
#         return predicted

class Predictor(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim)
        )
    def forward(self, x):
        return self.net(x)
