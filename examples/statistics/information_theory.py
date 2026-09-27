"""Entropy, cross-entropy, and KL divergence from post 64 (bits)."""

import numpy as np


def _distribution(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
            or np.any(values < 0) or not np.isclose(values.sum(), 1.0)):
        raise ValueError("expected a finite 1-D probability distribution summing to one")
    return values


def entropy(p: np.ndarray) -> float:
    p = _distribution(p)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _distribution(p), _distribution(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same events")
    positive = p > 0
    if np.any(q[positive] == 0):
        return float("inf")
    return float(-np.sum(p[positive] * np.log2(q[positive])))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return cross_entropy(p, q) - entropy(p)


if __name__ == "__main__":
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    print(f"H(P) = {entropy(p):.6f} bits")
    print(f"H(P,Q) = {cross_entropy(p, q):.6f} bits")
    print(f"KL(P||Q) = {kl_divergence(p, q):.6f} bits")
