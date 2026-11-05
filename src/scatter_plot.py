"""
相関が強い2科目を散布図で表示する
- 全科目のペアの相関係数 r を計算する
- |r| が大きい上位ペアを寮で色分けして描く
使い方: scatter_plot.py dataset.csv
"""

import sys
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from load import load
from stats42 import ft_corr

HOUSE = "Hogwarts House"
HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
TOP = 6
COLS = 3

Ranking = list[tuple[float, str, str]]


def parse_args() -> str:
    """
    return データセットのパス (使い方が違えば終了)
    """
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        sys.exit(1)
    return sys.argv[1]


def load_scores(path: str) -> pd.DataFrame:
    """
    return 寮と数値科目だけの DataFrame
    """
    df = load(path).dropna(subset=[HOUSE])
    courses = [c for c in df.select_dtypes(include="number").columns
               if c != "Index"]
    return df[[HOUSE, *courses]]


def rank_pairs(df: pd.DataFrame) -> Ranking:
    """
    return [(r, 科目a, 科目b)] |r| の大きい順
    """
    courses = [c for c in df.columns if c != HOUSE]
    ranking = []
    for a, b in combinations(courses, 2):
        pair = df[[a, b]].dropna()  # keep the students who have both
        ranking.append((ft_corr(pair[a].tolist(), pair[b].tolist()), a, b))
    return sorted(ranking, key=lambda t: -abs(t[0]))


def print_ranking(ranking: Ranking) -> None:
    print("correlation (|r| close to 1 = similar)")
    for r, a, b in ranking[:TOP]:
        print(f"  {r:+.3f}  {a} / {b}")


def draw(df: pd.DataFrame, ranking: Ranking) -> Figure:
    """
    return 上位ペアの散布図 (寮で色分け)
    """
    top = ranking[:TOP]
    rows = (len(top) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 4.5 * rows))
    flat = list(axes.flatten())
    for ax, (r, a, b) in zip(flat, top):
        pair = df[[HOUSE, a, b]].dropna()
        for h, g in pair.groupby(HOUSE):
            ax.scatter(g[a], g[b], s=8, alpha=0.5,
                       color=HOUSE_COLORS[str(h)], label=h)
        ax.set_xlabel(a, fontsize=8)
        ax.set_ylabel(b, fontsize=8)
        ax.set_title(f"r = {r:+.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat[len(top):]:
        ax.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10)
    fig.suptitle("Most correlated pairs of courses "
                 "(left to right, top to bottom: strongest first)")
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_scores(parse_args())
    ranking = rank_pairs(df)
    print_ranking(ranking)
    draw(df, ranking)
    plt.show()


if __name__ == "__main__":
    main()
