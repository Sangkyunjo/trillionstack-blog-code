"""AR(2) stability and impulse response from the difference-equation post."""

import numpy as np


def impulse_response(phi1: float, phi2: float, periods: int) -> np.ndarray:
    """Return the first `periods` coefficients of 1/(1-phi1 L-phi2 L²)."""
    if periods < 1:
        raise ValueError("periods must be positive")
    weights = np.zeros(periods, dtype=float)
    weights[0] = 1.0
    if periods > 1:
        weights[1] = phi1
    for j in range(2, periods):
        weights[j] = phi1 * weights[j - 1] + phi2 * weights[j - 2]
    return weights


def companion_eigenvalues(phi1: float, phi2: float) -> np.ndarray:
    companion = np.array([[phi1, phi2], [1.0, 0.0]])
    return np.linalg.eigvals(companion)


if __name__ == "__main__":
    phi1, phi2 = 0.5, 0.2
    eigenvalues = companion_eigenvalues(phi1, phi2)
    weights = impulse_response(phi1, phi2, 8)
    assert np.all(np.abs(eigenvalues) < 1)
    print("companion eigenvalues:", np.round(eigenvalues, 6))
    print("impulse response:", np.round(weights, 6))
