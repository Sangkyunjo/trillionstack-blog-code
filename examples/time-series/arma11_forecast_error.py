"""ARMA(1,1) psi weights and multi-step forecast error variance."""

import numpy as np


def psi_weights(phi: float, theta: float, horizon: int) -> np.ndarray:
    """Return psi_0 ... psi_(horizon-1) for (1+theta L)/(1-phi L)."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    weights = np.empty(horizon, dtype=float)
    weights[0] = 1.0
    for j in range(1, horizon):
        weights[j] = (phi + theta) * phi ** (j - 1)
    return weights


def forecast_error_variance(phi: float, theta: float, innovation_variance: float, horizon: int) -> float:
    if innovation_variance < 0:
        raise ValueError("innovation_variance must be nonnegative")
    weights = psi_weights(phi, theta, horizon)
    return float(innovation_variance * np.dot(weights, weights))


if __name__ == "__main__":
    phi, theta, variance = 0.55, 0.30, 1.4**2
    for horizon in (1, 2, 6):
        print(f"h={horizon}: {forecast_error_variance(phi, theta, variance, horizon):.6f}")
