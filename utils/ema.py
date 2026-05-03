"""Exponential Moving Average (EMA) update for the target encoder."""

import math
import torch.nn as nn


class EMAUpdater:
    """Manages EMA updates of a target network from an online network.

    Uses a cosine annealing schedule for the decay parameter τ:
        τ(step) = τ_end - (τ_end - τ_start) * (1 + cos(π * step / total_steps)) / 2

    This starts with τ_start (faster tracking) and anneals toward τ_end
    (slower tracking), matching the BYOL/DINO convention.
    """

    def __init__(
        self,
        tau_start: float = 0.996,
        tau_end: float = 0.9999,
        total_steps: int = 10000,
    ):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.current_step = 0

    @property
    def tau(self) -> float:
        """Current EMA decay value."""
        if self.total_steps <= 0:
            return self.tau_end
        progress = min(self.current_step / self.total_steps, 1.0)
        # Cosine annealing from tau_start to tau_end
        return self.tau_end - (self.tau_end - self.tau_start) * (1 + math.cos(math.pi * progress)) / 2

    @staticmethod
    def init_target(online: nn.Module, target: nn.Module) -> None:
        """Copy online network parameters to target network (no grad)."""
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.copy_(p_online.data)
            p_target.requires_grad_(False)

    def update(self, online: nn.Module, target: nn.Module) -> float:
        """Perform one EMA update step.

        θ_target = τ · θ_target + (1 - τ) · θ_online

        Returns:
            The τ value used for this update.
        """
        tau = self.tau
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)
        self.current_step += 1
        return tau
