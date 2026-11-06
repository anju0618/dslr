"""
one-vs-all のロジスティック回帰を学習して、重みを保存する
使い方: logreg_train.py dataset_train.csv
"""

import sys

import numpy as np
import pandas as pd

from load import load
from stats42 import ft_dropna, ft_mean, ft_std

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
    raise NotImplementedError


def make_labels(df: pd.DataFrame, house: str) -> np.ndarray:
    """
    return y (house の生徒は 1、それ以外は 0)
    """
    raise NotImplementedError


def train(X: np.ndarray, df: pd.DataFrame) -> dict[str, list[float]]:
    """
    return {寮名: θ}
    """
    raise NotImplementedError


def save_weights(
        path: str,
        thetas: dict[str, list[float]],
        means: dict[str, float],
        stds: dict[str, float]
        ) -> None:
    """
    θ、mean、std、科目の順番を保存する
    """
    raise NotImplementedError


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset_train.csv", file=sys.stderr)
        sys.exit(1)

    df = load(sys.argv[1])
    # TODO: compute_scaler → build_matrix → train → save_weights


if __name__ == "__main__":
    main()
