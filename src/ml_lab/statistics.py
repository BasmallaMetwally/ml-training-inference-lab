"""Seed-level descriptive and paired inferential statistics."""

import itertools

import numpy as np
from scipy import stats


def summarize(values: list[float] | np.ndarray) -> dict:
    """Student-t interval for seed means; not a population-data uncertainty claim."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Need at least two finite, one-dimensional observations")
    spread = float(a.std(ddof=1))
    margin = float(stats.t.ppf(0.975, len(a) - 1) * spread / np.sqrt(len(a)))
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "std": spread,
        "ci95": [float(a.mean() - margin), float(a.mean() + margin)],
    }


def paired_comparison(first: list[float] | np.ndarray, second: list[float] | np.ndarray) -> dict:
    """Exact two-sided sign-flip test, conditional on exchangeable paired differences."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not 2 <= len(a) <= 20:
        raise ValueError("Require 2 to 20 matched pairs")
    differences = b - a
    result = summarize(differences)
    null = np.array(
        [
            np.mean(differences * np.asarray(signs))
            for signs in itertools.product([-1, 1], repeat=len(a))
        ]
    )
    result["exact_sign_flip_p_two_sided"] = float(
        np.mean(np.abs(null) >= abs(differences.mean()) - 1e-15)
    )
    result["differences"] = differences.tolist()
    result["assumption"] = (
        "Independent seed pairs; signs exchangeable under the null; fixed dataset split"
    )
    return result
