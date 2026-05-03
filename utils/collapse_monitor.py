"""Collapse monitoring for JEPA pretraining.

Tracks the standard deviation of user embeddings on a fixed eval batch
to detect representation collapse early.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class CollapseMonitor:
    """Monitors embedding collapse during JEPA pretraining.

    Computes the mean std-dev across embedding dimensions on a fixed
    eval batch. If it drops below the threshold, emits a warning.
    Also tracks per-dimension statistics for deeper analysis.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.history: list[dict] = []

    @torch.no_grad()
    def check(
        self,
        embeddings: torch.Tensor,
        step: int,
    ) -> dict:
        """Check for collapse on a batch of embeddings.

        Args:
            embeddings: (batch, dim) — user/item embeddings to monitor.
            step: Current training step.

        Returns:
            Dict with metrics: mean_std, min_std, max_std, collapsed (bool).
        """
        # Per-dimension std across the batch
        per_dim_std = embeddings.std(dim=0)
        mean_std = per_dim_std.mean().item()
        min_std = per_dim_std.min().item()
        max_std = per_dim_std.max().item()
        num_dead = (per_dim_std < 1e-6).sum().item()

        collapsed = mean_std < self.threshold

        metrics = {
            "step": step,
            "mean_std": mean_std,
            "min_std": min_std,
            "max_std": max_std,
            "num_dead_dims": num_dead,
            "collapsed": collapsed,
        }
        self.history.append(metrics)

        if collapsed:
            logger.warning(
                f"⚠️  COLLAPSE DETECTED at step {step}: mean_std={mean_std:.6f} "
                f"< threshold={self.threshold}. "
                f"Dead dims: {num_dead}/{embeddings.shape[1]}"
            )
        else:
            logger.info(
                f"Collapse check step {step}: mean_std={mean_std:.4f}, "
                f"min_std={min_std:.4f}, dead_dims={num_dead}"
            )

        return metrics

    def get_trend(self, last_n: int = 10) -> str:
        """Get a summary of the recent std-dev trend."""
        if len(self.history) < 2:
            return "insufficient_data"
        recent = self.history[-last_n:]
        stds = [m["mean_std"] for m in recent]
        if stds[-1] < stds[0] * 0.5:
            return "declining_fast"
        elif stds[-1] < stds[0]:
            return "declining_slow"
        else:
            return "stable"
