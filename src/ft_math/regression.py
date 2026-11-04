"""
linear regression (from ft_linear_regression)
price = θ0 + θ1 * km

cost = (1 / 2m) * Σ(予測_i - 実際_i)²

TODO: dslr needs logistic regression (sigmoid + log loss, one-vs-all),
not linear regression. Kept as-is for now as a reference for the
gradient descent loop structure; predict/cost will be replaced.
"""

import numpy as np


def ft_predict(t0: float, t1: float, x: np.ndarray) -> np.ndarray:
    """
    ex: price = 8000 - 0.02 * 50000
        (θ0 = 8000, θ1 = - 0.02)
    return t0 + t1 * x
    """
    return t0 + t1 * x


def ft_cost(t0: float, t1: float, x: np.ndarray, y: np.ndarray) -> float:
    """
    return (1 / 2m) * Σ(予測 - 実際)²
    """
    # 残差平方和RSS
    predict = ft_predict(t0, t1, x)
    rss = ((predict - y) ** 2).sum()

    cost = rss / (2 * len(x))
    return float(cost)


def ft_gradient_descent(
        x: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        max_epochs: int = 10000,
        tol: float = 1e-9
        ) -> tuple[float, float, list[float]]:
    """
    return (t0, t1, cost history)
    stop when max_epochs is reached or cost improvement < tol
    """
    t0 = 0.0
    t1 = 0.0
    m = len(x)
    history: list[float] = [ft_cost(t0, t1, x, y)]

    for _ in range(max_epochs):
        err = ft_predict(t0, t1, x) - y         # 予測 - 実際(配列)
        grad0 = err.sum() / m                   # θ0 の傾き 誤差平均
        grad1 = (err * x).sum() / m             # θ1 の傾き

        t0 = t0 - lr * grad0
        t1 = t1 - lr * grad1

        history.append(ft_cost(t0, t1, x, y))
        if abs(history[-2] - history[-1]) < tol:
            break

    return t0, t1, history
