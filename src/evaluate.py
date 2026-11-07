"""
保存した重みで予測して、正解と比べて正答率を出す
同じデータで scikit-learn の LogisticRegression とも比較する
使い方: evaluate.py dataset_train.csv weights.json
"""

import sys

import pandas as pd
from sklearn.linear_model import (  # type: ignore[import-untyped]
    LogisticRegression
)

from load import load
from logreg_predict import load_weights, predict
from logreg_train import HOUSE, HOUSES, build_matrix


def report(name: str, predicted: list[str], answers: list[str]) -> None:
    """
    全体と寮ごとの正答率を表示する
    """
    total = len(answers)
    correct = 0
    house_total: dict[str, int] = {}
    house_correct: dict[str, int] = {}
    for house in HOUSES:
        house_total[house] = 0
        house_correct[house] = 0

    for pred, ans in zip(predicted, answers):
        house_total[ans] += 1
        if pred == ans:
            correct += 1
            house_correct[ans] += 1

    print(f"[{name}] accuracy: {correct / total * 100:.2f}% "
          f"({correct}/{total})")
    for house in HOUSES:
        n = house_total[house]
        ok = house_correct[house]
        print(f"  {house:11s} {ok / n * 100:6.2f}% ({ok}/{n})")


def sklearn_predict(df: pd.DataFrame, weights: dict) -> list[str]:
    """
    return scikit-learn で学習して予測した寮名 (同じ標準化、正則化なし)
    """
    X = build_matrix(df, weights["means"], weights["stds"])[:, 1:]
    model = LogisticRegression(penalty=None, max_iter=10000)
    model.fit(X, df[HOUSE].tolist())
    return model.predict(X).tolist()


def main() -> None:
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} dataset_train.csv weights.json",
            file=sys.stderr
        )
        sys.exit(1)

    df = load(sys.argv[1])
    if HOUSE not in df.columns or df[HOUSE].isna().any():
        print(f"Error: {HOUSE} column has no answers", file=sys.stderr)
        sys.exit(1)

    weights = load_weights(sys.argv[2])
    predicted = predict(df, weights)
    answers = df[HOUSE].tolist()

    report("ft", predicted, answers)
    report("sklearn", sklearn_predict(df, weights), answers)


if __name__ == "__main__":
    main()
