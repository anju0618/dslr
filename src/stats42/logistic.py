"""
ロジスティック回帰 (one-vs-all)
X は (m, n) で先頭列は 1 / y は 0/1 / theta は (n,)
- ft_sigmoid: シグモイド関数
- ft_predict: 予測確率
- ft_cost: 損失 (クロスエントロピー)
- ft_gradient: 損失の勾配
- ft_gradient_descent: バッチ勾配降下法
- ft_sgd: 確率的勾配降下法
- ft_minibatch_gd: ミニバッチ勾配降下法
"""

import numpy as np


def ft_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    return g(z) = 1 / (1 + e^(−z))
    """
    # clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def ft_predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    return h(x) = g(θᵀx)  (y = 1 の確率)
    """
    return ft_sigmoid(X @ theta)


def ft_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    return J(θ) = −(1/m) Σ [y·log(h) + (1 − y)·log(1 − h)]
    """
    h = np.clip(ft_predict(theta, X), 1e-15, 1 - 1e-15)
    loss = y * np.log(h) + (1 - y) * np.log(1 - h)
    return float(-loss.sum() / len(y))


def ft_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    return ∇J = (1/m) Xᵀ (h − y)
    """
    return X.T @ (ft_predict(theta, X) - y) / len(y)


def ft_gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        max_epochs: int = 10000,
        tol: float = 1e-9
        ) -> tuple[np.ndarray, list[float]]:
    """
    return (theta, 損失の履歴)
    バッチ勾配降下法: θ ← θ − lr · ∇J(θ)
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
    """
    return (theta, 損失の履歴)
    確率的勾配降下法: θ ← θ − lr · (hᵢ − yᵢ) · xᵢ
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
    """
    return (theta, 損失の履歴)
    ミニバッチ勾配降下法: θ ← θ − lr · (1/b) Σ_B (hᵢ − yᵢ) · xᵢ
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
