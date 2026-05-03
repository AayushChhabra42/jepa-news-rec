"""Item encoder: MiniLM projection + category/subcategory embeddings + entity flags → 128d."""

import torch
import torch.nn as nn
import numpy as np


class ItemEncoder(nn.Module):
    """Encodes news articles into a fixed-size embedding.

    Combines:
      1. Pre-computed MiniLM text embedding (384d) → linear projection → 128d
      2. Category embedding (16d)
      3. Subcategory embedding (16d)
      4. Entity presence flag (1d)
      5. Final projection: concat → linear → item_embed_dim

    The text embeddings are loaded from pre-computed numpy arrays to avoid
    running MiniLM at training time.
    """

    def __init__(
        self,
        text_embeddings: np.ndarray,
        num_categories: int = 19,       # includes PAD=0
        num_subcategories: int = 300,
        cat_embed_dim: int = 16,
        subcat_embed_dim: int = 16,
        text_dim: int = 384,
        item_embed_dim: int = 128,
    ):
        super().__init__()

        # Pre-computed text embeddings (frozen by default, unfrozen during JEPA)
        self.text_embed = nn.Embedding.from_pretrained(
            torch.from_numpy(text_embeddings), freeze=False, padding_idx=0
        )

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, item_embed_dim),
            nn.LayerNorm(item_embed_dim),
            nn.GELU(),
        )

        # Category / subcategory embeddings
        self.cat_embed = nn.Embedding(num_categories, cat_embed_dim, padding_idx=0)
        self.subcat_embed = nn.Embedding(num_subcategories, subcat_embed_dim, padding_idx=0)

        # Final fusion
        fused_dim = item_embed_dim + cat_embed_dim + subcat_embed_dim + 1  # +1 for entity flag
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, item_embed_dim),
            nn.LayerNorm(item_embed_dim),
            nn.GELU(),
        )

        self.item_embed_dim = item_embed_dim

    def forward(
        self,
        news_ids: torch.Tensor,
        cat_ids: torch.Tensor,
        subcat_ids: torch.Tensor,
        entity_flags: torch.Tensor,
    ) -> torch.Tensor:
        """Encode articles.

        Args:
            news_ids: (...) — article indices into text_embed.
            cat_ids: (...) — category indices.
            subcat_ids: (...) — subcategory indices.
            entity_flags: (...) — float, 1.0 if has entities.

        Returns:
            (..., item_embed_dim) — article embeddings.
        """
        # Text
        text = self.text_embed(news_ids)           # (..., 384)
        text = self.text_proj(text)                 # (..., 128)

        # Categorical
        cat = self.cat_embed(cat_ids)               # (..., 16)
        subcat = self.subcat_embed(subcat_ids)      # (..., 16)

        # Entity flag — expand to match batch dims
        ent = entity_flags.unsqueeze(-1)            # (..., 1)

        # Fuse
        fused = torch.cat([text, cat, subcat, ent], dim=-1)  # (..., 161)
        return self.fusion(fused)                   # (..., 128)

    def get_all_embeddings(
        self,
        cat_ids_all: torch.Tensor,
        subcat_ids_all: torch.Tensor,
        entity_flags_all: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embeddings for ALL articles at once (for FAISS indexing).

        Args:
            cat_ids_all: (num_articles,)
            subcat_ids_all: (num_articles,)
            entity_flags_all: (num_articles,)

        Returns:
            (num_articles, item_embed_dim)
        """
        num_articles = cat_ids_all.shape[0]
        all_ids = torch.arange(num_articles, device=cat_ids_all.device)
        return self.forward(all_ids, cat_ids_all, subcat_ids_all, entity_flags_all)
