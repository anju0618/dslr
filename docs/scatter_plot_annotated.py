# 解説用コピー: src/scatter_plot.py にコメントを足したもの (コードは同一)
#
# 課題の問い: 「似ている2つの特徴量はどれか」
# 「似ている」= 片方の点が分かればもう片方も分かる = 相関が強い
# 流れ (main): parse_args -> load_scores -> rank_pairs
#              -> print_ranking -> draw -> plt.show
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
from stats42 import ft_corr  # ピアソンの相関係数 (自作)

HOUSE = "Hogwarts House"
HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
TOP = 6   # 散布図にする上位ペアの数
COLS = 3  # 図を何列に並べるか

# 型エイリアス (動作には影響しない)
# Ranking: [(r, 科目a, 科目b), ...]
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
    # 寮が無い行を捨てて 数値の列 (Index 以外) だけを科目として残す
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
    # combinations(courses, 2): 科目から2つ選ぶ全組み合わせ
    # 13科目なら 13*12/2 = 78通り (a,b) と (b,a) は同じ扱いで1回だけ
    for a, b in combinations(courses, 2):
        # 2科目のどちらかが欠損の生徒を落とす
        # 同じ行の値が同じ生徒なので 残った行は2科目の点がそろっている
        pair = df[[a, b]].dropna()  # keep the students who have both
        # ft_corr は NaN なし・同じ長さのリストが前提なので tolist() で渡す
        ranking.append((ft_corr(pair[a].tolist(), pair[b].tolist()), a, b))
    # key=lambda t: -abs(t[0]) で |r| の大きい順にする
    # abs を使うのは r = -1 (逆向きの完全な直線) も「似ている」に入れるため
    return sorted(ranking, key=lambda t: -abs(t[0]))


def print_ranking(ranking: Ranking) -> None:
    print("correlation (|r| close to 1 = similar)")
    for r, a, b in ranking[:TOP]:  # 上位 TOP 個だけ表示
        print(f"  {r:+.3f}  {a} / {b}")  # :+ は符号を必ず付ける指定


def draw(df: pd.DataFrame, ranking: Ranking) -> Figure:
    """
    return 上位ペアの散布図 (寮で色分け)
    """
    top = ranking[:TOP]
    rows = (len(top) + COLS - 1) // COLS  # 切り上げ
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 4.5 * rows))
    flat = list(axes.flatten())
    for ax, (r, a, b) in zip(flat, top):
        # 寮の列と2科目を取り出し 欠損のある行を落とす
        pair = df[[HOUSE, a, b]].dropna()
        # 寮ごとに色を変えて 寮の数だけ点を打つ
        for h, g in pair.groupby(HOUSE):
            ax.scatter(g[a], g[b], s=8, alpha=0.5,
                       color=HOUSE_COLORS[str(h)], label=h)
        ax.set_xlabel(a, fontsize=8)  # 横軸: 科目a
        ax.set_ylabel(b, fontsize=8)  # 縦軸: 科目b
        ax.set_title(f"r = {r:+.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat[len(top):]:  # 余った枠は消す
        ax.axis("off")
    # 凡例は全グラフ共通なので 最初のグラフから取って1つだけ出す
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10)
    fig.suptitle("Most correlated pairs of courses "
                 "(left to right, top to bottom: strongest first)")
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_scores(parse_args())   # 寮 + 科目 の表
    ranking = rank_pairs(df)         # 全78ペアを |r| の大きい順に
    print_ranking(ranking)
    draw(df, ranking)
    plt.show()


if __name__ == "__main__":
    main()
