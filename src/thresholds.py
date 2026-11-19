"""Empirical benign-score budget with conservative treatment of ties.

Scores must be finite probabilities. Decisions use p >= threshold. The returned
threshold is representable in the score dtype, including a value ABOVE 1 when
probabilities saturate at 1. This boundary means no alerts, not a clipped 1.
No population-level FPR guarantee is implied by an empirical budget.
"""
import numpy as np

THRESHOLD_VERSION = "empirical-budget-v2-positive-infinity"


def benign_budget_threshold(benign_scores, alpha=0.01):
    scores = np.asarray(benign_scores)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("Expected a nonempty one-dimensional benign score array")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if scores.dtype.kind != "f":
        scores = scores.astype(np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    if alpha == 1:
        return 0.0
    k = int(np.floor(alpha * scores.size))
    boundary = np.sort(scores)[scores.size - k - 1]
    threshold = np.nextafter(boundary, np.array(np.inf, dtype=scores.dtype))
    threshold = float(threshold)
    # Tied observations are excluded together; random tie-breaking is not used.
    if np.count_nonzero(scores >= threshold) > k:
        raise RuntimeError("Numerical boundary failed to enforce the empirical budget")
    return threshold


def labelled_budget_threshold(y, scores, alpha=0.01):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.shape != scores.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching one-dimensional binary labels and scores")
    return benign_budget_threshold(scores[y == 0], alpha)
