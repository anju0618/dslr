"""
全科目の組み合わせを散布図行列で表示する
- 対角線は科目のヒストグラム
- それ以外は2科目の散布図 (寮で色分け)
- 寮が分かれる科目を選ぶために使う
使い方: pair_plot.py dataset.csv
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from load import load

HOUSE = "Hogwarts House"
HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
BINS = 20
CELL = 1.8


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


def draw(df: pd.DataFrame) -> Figure:
    """
    return 散布図行列 (対角線: ヒストグラム 他: 散布図)
    """
    courses = [c for c in df.columns if c != HOUSE]
    houses = {str(h): g for h, g in df.groupby(HOUSE)}
    n = len(courses)
    fig, axes = plt.subplots(n, n, figsize=(CELL * n, CELL * n))
    for i, row in enumerate(courses):
        for j, col in enumerate(courses):
            ax = axes[i][j]
            for h, g in houses.items():
                if i == j:
                    ax.hist(g[col].dropna(), bins=BINS, alpha=0.5,
                            color=HOUSE_COLORS[h])
                else:
                    ax.scatter(g[col], g[row], s=2, alpha=0.5,
                               color=HOUSE_COLORS[h])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row, fontsize=6, rotation=0, ha="right")
            if i == n - 1:
                ax.set_xlabel(col, fontsize=6, rotation=45, ha="right")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c)
               for c in HOUSE_COLORS.values()]
    fig.legend(handles, list(HOUSE_COLORS), loc="upper right", fontsize=10)
    fig.suptitle("Pair plot (diagonal: histogram, others: scatter plot)")
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_scores(parse_args())
    draw(df)
    plt.show()


if __name__ == "__main__":
    main()
