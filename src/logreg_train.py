"""
one-vs-all のロジスティック回帰を学習して、重みを保存する
使い方: logreg_train.py dataset_train.csv [--algo batch|sgd|minibatch]
"""

import argparse
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from load import load
from stats42 import (
    ft_dropna, ft_gradient_descent, ft_mean, ft_minibatch_gd, ft_sgd, ft_std
)

HOUSE = "Hogwarts House"
HOUSES = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")
FEATURES: list[str] = [
    "Charms",
    "Astronomy",
    "Herbology",
    "Ancient Runes",
    "Divination",
    "Muggle Studies"
]
WEIGHTS_PATH = "weights.json"
ALGOS: dict[str, Callable[..., tuple[np.ndarray, list[float]]]] = {
    "batch": ft_gradient_descent,
    "sgd": ft_sgd,
    "minibatch": ft_minibatch_gd,
}


def compute_scaler(
        df: pd.DataFrame
        ) -> tuple[dict[str, float], dict[str, float]]:
    """
    return (科目ごとの mean, 科目ごとの std)
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for feature in FEATURES:
        scores = ft_dropna(df[feature].tolist())
        means[feature] = ft_mean(scores)
        stds[feature] = ft_std(scores)

    return (means, stds)


def build_matrix(
        df: pd.DataFrame,
        means: dict[str, float],
        stds: dict[str, float]
        ) -> np.ndarray:
    """
    return X (標準化 → NaN を 0 → 先頭に 1 の列)
    """
    columns: list[list[float]] = [[1.0] * len(df)]
    for feature in FEATURES:
        lis = []
        for score in df[feature].tolist():
            z = (score - means[feature]) / stds[feature]
            lis.append(0.0 if z != z else z)
        columns.append(lis)

    return np.column_stack(columns)


def make_labels(df: pd.DataFrame, house: str) -> np.ndarray:
    """
    return y (house の生徒は 1、それ以外は 0)
    """
    labels = []
    for h in df[HOUSE].tolist():
        if h == house:
            labels.append(1.0)
        else:
            labels.append(0.0)
    return np.array(labels)


def train(
        X: np.ndarray,
        df: pd.DataFrame,
        algo: str = "batch"
        ) -> dict[str, list[float]]:
    """
    return {寮名: θ}  (algo: batch / sgd / minibatch)
    """
    thetas: dict[str, list[float]] = {}

    for house in HOUSES:
        y = make_labels(df, house)
        theta, history = ALGOS[algo](X, y)
        print(f"{house:11s} cost {history[0]:.4f} -> {history[-1]:.4f}")
        thetas[house] = theta.tolist()

    return thetas


def save_weights(
        path: str,
        thetas: dict[str, list[float]],
        means: dict[str, float],
        stds: dict[str, float]
        ) -> None:
    """
    θ、mean、std、科目の順番を保存する
    """
    data = {
        "features": FEATURES,
        "means": means,
        "stds": stds,
        "thetas": thetas,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("dataset", help="dataset_train.csv")
    parser.add_argument("--algo", choices=ALGOS, default="batch")
    args = parser.parse_args()

    df = load(args.dataset)
    means, stds = compute_scaler(df)

    X = build_matrix(df, means, stds)
    thetas = train(X, df, args.algo)
    save_weights(WEIGHTS_PATH, thetas, means, stds)


if __name__ == "__main__":
    main()
