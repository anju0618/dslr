"""Logistic regression (one-vs-all).

X is (m, n) with a leading column of 1s, y is 0/1, theta is (n,).
"""

import numpy as np


def ft_sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    Formula:
        g(z) = 1 / (1 + e^(−z))

    Args:
        z: array.

    Returns:
        Values in (0, 1).
    """
    # clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def ft_predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return P(y = 1) for each sample.

    Formula:
        h(x) = g(θᵀx)

    Args:
        theta: weights (n,).
        X: features (m, n).

    Returns:
        Probabilities (m,).
    """
    return ft_sigmoid(X @ theta)


def ft_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Return the cross-entropy loss.

    Formula:
        J(θ) = −(1/m) Σ [y·log(h) + (1 − y)·log(1 − h)]

    Args:
        theta: weights (n,).
        X: features (m, n).
        y: 0/1 labels (m,).

    Returns:
        Loss.
    """
    h = np.clip(ft_predict(theta, X), 1e-15, 1 - 1e-15)
    loss = y * np.log(h) + (1 - y) * np.log(1 - h)
    return float(-loss.sum() / len(y))


def ft_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the gradient of the loss.

    Formula:
        ∇J = (1/m) Xᵀ (h − y)

    Args:
        theta: weights (n,).
        X: features (m, n).
        y: 0/1 labels (m,).

    Returns:
        Gradient (n,).
    """
    return X.T @ (ft_predict(theta, X) - y) / len(y)


def ft_gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        max_epochs: int = 10000,
        tol: float = 1e-9
        ) -> tuple[np.ndarray, list[float]]:
    """Train with batch gradient descent.

    Formula:
        θ ← θ − lr · ∇J(θ)

    Args:
        X: features (m, n).
        y: 0/1 labels (m,).
        lr: learning rate.
        max_epochs: max epochs.
        tol: stop if the loss gain is below this.

    Returns:
        (theta, loss history).
    """
    theta = np.zeros(X.shape[1])
    history: list[float] = [ft_cost(theta, X, y)]
    for _ in range(max_epochs):
        theta = theta - lr * ft_gradient(theta, X, y)
        history.append(ft_cost(theta, X, y))
        if abs(history[-2] - history[-1]) < tol:
            break
    return theta, history


def ft_sgd(
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        max_epochs: int = 100,
        seed: int = 0
        ) -> tuple[np.ndarray, list[float]]:
    """Train with stochastic gradient descent (bonus).

    Formula:
        θ ← θ − lr · (hᵢ − yᵢ) · xᵢ

    Args:
        X: features (m, n).
        y: 0/1 labels (m,).
        lr: learning rate.
        max_epochs: epochs.
        seed: shuffle seed.

    Returns:
        (theta, loss history).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros(X.shape[1])
    history: list[float] = [ft_cost(theta, X, y)]
    for _ in range(max_epochs):
        for i in rng.permutation(len(y)):
            err = ft_sigmoid(X[i] @ theta) - y[i]
            theta = theta - lr * err * X[i]
        history.append(ft_cost(theta, X, y))
    return theta, history


def ft_minibatch_gd(
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        batch_size: int = 32,
        max_epochs: int = 100,
        seed: int = 0
        ) -> tuple[np.ndarray, list[float]]:
    """Train with mini-batch gradient descent (bonus).

    Formula:
        θ ← θ − lr · (1/b) Σ_B (hᵢ − yᵢ) · xᵢ

    Args:
        X: features (m, n).
        y: 0/1 labels (m,).
        lr: learning rate.
        batch_size: b.
        max_epochs: epochs.
        seed: shuffle seed.

    Returns:
        (theta, loss history).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros(X.shape[1])
    history: list[float] = [ft_cost(theta, X, y)]
    for _ in range(max_epochs):
        idx = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            b = idx[start:start + batch_size]
            theta = theta - lr * ft_gradient(theta, X[b], y[b])
        history.append(ft_cost(theta, X, y))
    return theta, history
