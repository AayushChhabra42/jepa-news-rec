"""VICReg-style variance and covariance regularisation terms.

Added as a proactive collapse prevention mechanism alongside EMA.
Reference: Bardes, Ponce, LeCun — "VICReg" (ICLR 2022).
"""

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Variance regularisation — hinge loss on per-dimension std.

    Penalises if the standard deviation of any embedding dimension
    across the batch falls below `gamma`.

    Args:
        z: (batch, dim) — batch of embeddings.
        gamma: Target minimum std per dimension.
        eps: Numerical stability.

    Returns:
        Scalar loss.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Covariance regularisation — penalise off-diagonal correlations.

    Encourages decorrelated embedding dimensions.

    Args:
        z: (batch, dim) — batch of embeddings.

    Returns:
        Scalar loss.
    """
    batch_size, dim = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)
    # Zero out diagonal (we don't penalise variance here)
    off_diag = cov - torch.diag(cov.diag())
    return (off_diag ** 2).sum() / dim


def vicreg_regularisation(
    z: torch.Tensor,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """Combined VICReg regularisation loss.

    Args:
        z: (batch, dim) — predictor output embeddings.
        lambda_var: Weight for variance term.
        lambda_cov: Weight for covariance term.

    Returns:
        (loss, metrics_dict) — scalar loss and breakdown for logging.
    """
    v_loss = variance_loss(z)
    c_loss = covariance_loss(z)
    total = lambda_var * v_loss + lambda_cov * c_loss
    return total, {"var_loss": v_loss.item(), "cov_loss": c_loss.item()}
