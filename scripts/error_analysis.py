"""Stage 7: Error analysis — collapse check, per-category breakdown, failure cases."""

import os
import sys
import pickle
import argparse
import logging

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_impression

logger = logging.getLogger(__name__)


def analyze_embeddings(checkpoint_path: str) -> dict:
    """Check for collapse in saved embeddings."""
    print("\n" + "=" * 60)
    print("  Embedding Analysis")
    print("=" * 60)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    results = {}

    if "collapse_history" in ckpt:
        history = ckpt["collapse_history"]
        print(f"  Collapse monitoring history: {len(history)} checkpoints")

        stds = [h["mean_std"] for h in history]
        print(f"  Std-dev range: {min(stds):.4f} — {max(stds):.4f}")
        print(f"  Final std-dev: {stds[-1]:.4f}")
        print(f"  Dead dims (final): {history[-1]['num_dead_dims']}")

        if stds[-1] < 0.05:
            print("  ⚠️  WARNING: Possible collapse detected!")
        else:
            print("  ✓ No collapse detected")

        results["collapse_history"] = history

    return results


def per_category_analysis(
    jepa_impressions: list[dict],
    xgb_impressions: list[dict],
    impression_categories: list[str],
) -> dict:
    """Compare JEPA vs XGBoost performance per category."""
    print("\n" + "=" * 60)
    print("  Per-Category Analysis")
    print("=" * 60)

    cat_jepa = defaultdict(list)
    cat_xgb = defaultdict(list)

    for imp, cat in zip(jepa_impressions, impression_categories):
        metrics = evaluate_impression(imp["y_true"], imp["y_score"])
        cat_jepa[cat].append(metrics)

    for imp, cat in zip(xgb_impressions, impression_categories):
        metrics = evaluate_impression(imp["y_true"], imp["y_score"])
        cat_xgb[cat].append(metrics)

    results = {}
    print(f"\n  {'Category':<20} {'JEPA AUC':>10} {'XGB AUC':>10} {'Δ AUC':>10} {'Count':>8}")
    print(f"  {'-'*58}")

    for cat in sorted(cat_jepa.keys()):
        j_metrics = cat_jepa.get(cat, [])
        x_metrics = cat_xgb.get(cat, [])

        if not j_metrics:
            continue

        j_auc = np.mean([m["auc"] for m in j_metrics])
        x_auc = np.mean([m["auc"] for m in x_metrics]) if x_metrics else 0

        delta = j_auc - x_auc
        sign = "+" if delta > 0 else ""

        results[cat] = {
            "jepa_auc": j_auc,
            "xgb_auc": x_auc,
            "delta": delta,
            "count": len(j_metrics),
        }

        winner = "🟢" if delta > 0.01 else ("🔴" if delta < -0.01 else "⚪")
        print(f"  {cat:<20} {j_auc:>10.4f} {x_auc:>10.4f} {sign}{delta:>9.4f} {len(j_metrics):>8d} {winner}")

    return results


def failure_case_analysis(
    jepa_impressions: list[dict],
    xgb_impressions: list[dict],
    top_k: int = 20,
) -> None:
    """Find impressions where XGB significantly outperforms JEPA."""
    print("\n" + "=" * 60)
    print("  Failure Case Analysis (XGB wins)")
    print("=" * 60)

    deltas = []
    for i, (j_imp, x_imp) in enumerate(zip(jepa_impressions, xgb_impressions)):
        j_m = evaluate_impression(j_imp["y_true"], j_imp["y_score"])
        x_m = evaluate_impression(x_imp["y_true"], x_imp["y_score"])
        deltas.append({
            "index": i,
            "jepa_auc": j_m["auc"],
            "xgb_auc": x_m["auc"],
            "delta": j_m["auc"] - x_m["auc"],
            "num_candidates": len(j_imp["y_true"]),
            "num_clicks": int(j_imp["y_true"].sum()),
        })

    # Sort by delta (JEPA - XGB), ascending = worst JEPA failures
    deltas.sort(key=lambda x: x["delta"])

    print(f"\n  Top {top_k} impressions where XGB beats JEPA:")
    print(f"  {'Idx':>6} {'JEPA':>8} {'XGB':>8} {'Δ':>8} {'Cands':>6} {'Clicks':>6}")
    print(f"  {'-'*44}")

    for d in deltas[:top_k]:
        print(
            f"  {d['index']:>6d} {d['jepa_auc']:>8.4f} {d['xgb_auc']:>8.4f} "
            f"{d['delta']:>+8.4f} {d['num_candidates']:>6d} {d['num_clicks']:>6d}"
        )

    # Summary statistics
    xgb_wins = sum(1 for d in deltas if d["delta"] < -0.01)
    jepa_wins = sum(1 for d in deltas if d["delta"] > 0.01)
    ties = len(deltas) - xgb_wins - jepa_wins

    print(f"\n  Summary:")
    print(f"    JEPA wins: {jepa_wins} ({100*jepa_wins/len(deltas):.1f}%)")
    print(f"    XGB wins:  {xgb_wins} ({100*xgb_wins/len(deltas):.1f}%)")
    print(f"    Ties:      {ties} ({100*ties/len(deltas):.1f}%)")


def run_error_analysis(
    processed_dir: str = "data/processed",
    jepa_checkpoint: str = "checkpoints/jepa_final.pt",
):
    """Run the full error analysis pipeline."""
    print("\n" + "#" * 60)
    print("  JEPA News Recommendation — Error Analysis")
    print("#" * 60)

    # 1. Embedding collapse check
    if os.path.exists(jepa_checkpoint):
        analyze_embeddings(jepa_checkpoint)
    else:
        print(f"  ⚠️  JEPA checkpoint not found at {jepa_checkpoint}")

    print("\n  Note: Per-category and failure case analysis require")
    print("  running evaluate.py first to generate impression-level scores.")
    print("  Use --model both to generate comparison data.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Error analysis")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--jepa-checkpoint", default="checkpoints/jepa_final.pt")
    args = parser.parse_args()

    run_error_analysis(
        processed_dir=args.processed_dir,
        jepa_checkpoint=args.jepa_checkpoint,
    )
