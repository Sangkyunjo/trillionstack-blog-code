"""OLS with a numerically stable least-squares solver, matching post 62."""

import numpy as np


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y must be same-length 1-D arrays with at least two observations")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain finite values")
    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("x must vary so the intercept and slope are identifiable")
    residuals = y - design @ coefficients
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    r_squared = float("nan") if total_sum_squares == 0 else 1 - float(residuals @ residuals) / total_sum_squares
    return coefficients, r_squared


if __name__ == "__main__":
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = np.array([3900.0, 6100.0, 7900.0, 10200.0, 12100.0])
    coefficients, r_squared = fit_line(x, y)
    print("intercept, slope:", coefficients)
    print("R-squared:", round(r_squared, 4))
