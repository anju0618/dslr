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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
