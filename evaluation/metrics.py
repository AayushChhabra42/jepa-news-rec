"""Evaluation metrics: AUC, MRR, nDCG@5, nDCG@10."""

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """Compute DCG@k."""
    scores = scores[:k]
    gains = 2 ** scores - 1
    discounts = np.log2(np.arange(len(scores)) + 2)
    return np.sum(gains / discounts)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute nDCG@k for a single impression."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    dcg = dcg_at_k(y_true_sorted, k)
    ideal_dcg = dcg_at_k(np.sort(y_true)[::-1], k)

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute MRR (Mean Reciprocal Rank) for a single impression."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    for rank, label in enumerate(y_true_sorted, start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC for a single impression."""
    if len(np.unique(y_true)) < 2:
        return 0.5  # Undefined — only one class present
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


def evaluate_impression(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics for a single impression.

    Args:
        y_true: (num_candidates,) — binary click labels.
        y_score: (num_candidates,) — predicted scores.

    Returns:
        Dict with AUC, MRR, nDCG@5, nDCG@10.
    """
    return {
        "auc": auc_score(y_true, y_score),
        "mrr": mrr(y_true, y_score),
        "ndcg@5": ndcg_at_k(y_true, y_score, 5),
        "ndcg@10": ndcg_at_k(y_true, y_score, 10),
    }


def evaluate_all(
    impressions: list[dict],
    history_lengths: list[int] | None = None,
    history_slices: list[int] | None = None,
) -> dict:
    """Evaluate over a list of impressions.

    Args:
        impressions: List of {"y_true": np.array, "y_score": np.array}.
        history_lengths: Optional list of user history lengths (for slicing).
        history_slices: Boundaries for user groups, e.g. [5, 20].

    Returns:
        Dict with overall metrics and per-group metrics.
    """
    if history_slices is None:
        history_slices = [5, 20]

    # Overall metrics
    all_metrics = [evaluate_impression(imp["y_true"], imp["y_score"]) for imp in impressions]

    results = {
        "overall": {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in ["auc", "mrr", "ndcg@5", "ndcg@10"]
        },
        "num_impressions": len(impressions),
    }

    # Per-group metrics (if history lengths provided)
    if history_lengths is not None:
        groups = {}
        boundaries = [0] + history_slices + [float("inf")]
        group_names = []
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            name = f"history_{lo}-{hi}" if hi != float("inf") else f"history_{lo}+"
            group_names.append(name)
            group_mask = [
                lo <= hl < hi for hl in history_lengths
            ]
            group_metrics = [m for m, mask in zip(all_metrics, group_mask) if mask]
            if group_metrics:
                groups[name] = {
                    metric: np.mean([m[metric] for m in group_metrics])
                    for metric in ["auc", "mrr", "ndcg@5", "ndcg@10"]
                }
                groups[name]["count"] = len(group_metrics)
            else:
                groups[name] = {"count": 0}

        results["groups"] = groups

    return results


def print_results(results: dict, model_name: str = "Model") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")

    overall = results["overall"]
    print(f"  AUC:      {overall['auc']:.4f}")
    print(f"  MRR:      {overall['mrr']:.4f}")
    print(f"  nDCG@5:   {overall['ndcg@5']:.4f}")
    print(f"  nDCG@10:  {overall['ndcg@10']:.4f}")
    print(f"  Impressions: {results['num_impressions']}")

    if "groups" in results:
        print(f"\n  {'Group':<20} {'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8} {'Count':>8}")
        print(f"  {'-'*52}")
        for name, metrics in results["groups"].items():
            if metrics["count"] > 0:
                print(
                    f"  {name:<20} {metrics['auc']:>8.4f} {metrics['mrr']:>8.4f} "
                    f"{metrics['ndcg@5']:>8.4f} {metrics['ndcg@10']:>8.4f} {metrics['count']:>8d}"
                )
            else:
                print(f"  {name:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8d}")

    print(f"{'='*60}\n")
