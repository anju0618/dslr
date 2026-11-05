"""
寮ごとの点数分布を科目別のヒストグラムで表示する
- 寮ごとの平均のばらつき (spread) で科目を並べる
- spread が小さい科目ほど 4寮が似ている
使い方: histogram.py dataset.csv
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from load import load
from stats42 import ft_mean, ft_std

HOUSE = "Hogwarts House"
HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
BINS = 20
COLS = 2

Scores = dict[str, list[float]]
Ranking = list[tuple[float, str]]


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


def house_scores(df: pd.DataFrame, course: str) -> Scores:
    """
    return {寮: 科目の点のリスト} (NaN 除去)
    """
    return {str(h): g.dropna().tolist()
            for h, g in df.groupby(HOUSE)[course]}


def spread(scores: Scores) -> float:
    """
    return std(寮ごとの平均) / std(全体)
    小さいほど4寮が似ている
    """
    means = [ft_mean(v) for v in scores.values()]
    values = [x for v in scores.values() for x in v]
    return ft_std(means) / ft_std(values)


def rank_courses(df: pd.DataFrame) -> Ranking:
    """
    return [(spread, 科目)] 小さい順
    """
    courses = [c for c in df.columns if c != HOUSE]
    return sorted((spread(house_scores(df, c)), c) for c in courses)


def print_ranking(ranking: Ranking) -> None:
    print("spread (small = homogeneous)")
    for s, c in ranking:
        print(f"  {s:.3f}  {c}")


def draw(df: pd.DataFrame, ranking: Ranking) -> Figure:
    """
    return 科目ごとに4寮を重ねたヒストグラム
    """
    rows = (len(ranking) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 3.2 * rows))
    flat = list(axes.flatten())
    for ax, (s, c) in zip(flat, ranking):
        for h, values in house_scores(df, c).items():
            ax.hist(values, bins=BINS, alpha=0.5, density=True,
                    color=HOUSE_COLORS[h], label=h)
        ax.set_title(f"{c}  (spread {s:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat[len(ranking):]:
        ax.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10)
    fig.suptitle("Score distribution per house "
                 "(left to right, top to bottom: most homogeneous first)")
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_scores(parse_args())
    ranking = rank_courses(df)
    print_ranking(ranking)
    draw(df, ranking)
    plt.show()


if __name__ == "__main__":
    main()
