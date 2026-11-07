"""
保存した重みで寮を予測して、houses.csv に書く
使い方: logreg_predict.py dataset_test.csv weights.json
"""

import json
import sys

import numpy as np
import pandas as pd

from load import load
from logreg_train import HOUSE, build_matrix
from stats42 import ft_predict

OUTPUT_PATH = "houses.csv"


def load_weights(path: str) -> dict:
    """
    return weights.json の中身 (エラーなら exit(1))
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def predict(df: pd.DataFrame, weights: dict) -> list[str]:
    """
    return 生徒ごとの寮名 (確率が最大の寮)
    """
    X = build_matrix(df, weights["means"], weights["stds"])
    houses = list(weights["thetas"].keys())
    columns = []
    for house in houses:
        theta = np.array(weights["thetas"][house])
        columns.append(ft_predict(theta, X))
    probs = np.column_stack(columns)

    result = []
    for i in probs.argmax(axis=1):
        result.append(houses[i])
    return result


def main() -> None:
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} dataset_test.csv weights.json",
            file=sys.stderr
        )
        sys.exit(1)

    df = load(sys.argv[1])
    weights = load_weights(sys.argv[2])
    result = pd.DataFrame({"Index": df["Index"], HOUSE: predict(df, weights)})
    result.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
