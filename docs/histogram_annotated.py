# 解説用コピー: src/histogram.py にコメントを足したもの (コードは同一)
#
# 課題の問い: 「どの科目で 4つの寮の点数分布が均質か」
# 流れ (main): parse_args -> load_scores -> rank_courses
#              -> print_ranking -> draw -> plt.show
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

HOUSE = "Hogwarts House"  # 寮の列の名前
# 寮ごとの色 (ホグワーツの寮カラー) 辞書のキーは寮の名前
HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
BINS = 20  # ヒストグラムの棒の数
COLS = 2   # 図を何列に並べるか

# 型エイリアス (長い型に名前を付けただけ 動作には影響しない)
# Scores : {寮: その寮の点のリスト}
# Ranking: [(spread, 科目), ...]
Scores = dict[str, list[float]]
Ranking = list[tuple[float, str]]


def parse_args() -> str:
    """
    return データセットのパス (使い方が違えば終了)
    """
    # sys.argv[0] はスクリプト名 引数はちょうど1個 (CSV のパス) だけ受け付ける
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} dataset.csv", file=sys.stderr)
        sys.exit(1)
    return sys.argv[1]


def load_scores(path: str) -> pd.DataFrame:
    """
    return 寮と数値科目だけの DataFrame
    """
    # 寮が書かれていない行は使えないので捨てる
    df = load(path).dropna(subset=[HOUSE])
    # 数値の列を科目とみなす (Index は科目ではないので除く)
    courses = [c for c in df.select_dtypes(include="number").columns
               if c != "Index"]
    # 名前や誕生日などの列を落として 寮 + 科目 だけにする
    return df[[HOUSE, *courses]]


def house_scores(df: pd.DataFrame, course: str) -> Scores:
    """
    return {寮: 科目の点のリスト} (NaN 除去)
    """
    # groupby で寮ごとに分けて 科目の列だけ取り出す
    # g.dropna() で欠損を除いて tolist() でリストにする
    # str(h) は groupby のキーの型を str にそろえるため (mypy 対策)
    return {str(h): g.dropna().tolist()
            for h, g in df.groupby(HOUSE)[course]}


def spread(scores: Scores) -> float:
    """
    return std(寮ごとの平均) / std(全体)
    小さいほど4寮が似ている
    """
    # 寮ごとの平均が どれだけ散らばっているかを数値にする
    # 全体のばらつきで割るのは 科目ごとの点数の単位の違いを消すため
    # (Arithmancy は 5万点台 Care of Magical Creatures は ±1 程度)
    #   小さい: 4寮の平均がほぼ同じ -> 寮を見分けられない科目
    #   大きい: 寮ごとに平均が違う   -> 寮を見分けられる科目
    means = [ft_mean(v) for v in scores.values()]      # 寮ごとの平均 (4個)
    values = [x for v in scores.values() for x in v]   # 全寮の点を1つに
    return ft_std(means) / ft_std(values)


def rank_courses(df: pd.DataFrame) -> Ranking:
    """
    return [(spread, 科目)] 小さい順
    """
    courses = [c for c in df.columns if c != HOUSE]
    # (spread, 科目名) のタプルは 先頭の spread から比べられるので
    # sorted にそのまま渡せば小さい順になる
    return sorted((spread(house_scores(df, c)), c) for c in courses)


def print_ranking(ranking: Ranking) -> None:
    print("spread (small = homogeneous)")
    for s, c in ranking:
        print(f"  {s:.3f}  {c}")


def draw(df: pd.DataFrame, ranking: Ranking) -> Figure:
    """
    return 科目ごとに4寮を重ねたヒストグラム
    """
    # 必要な行数 (切り上げ) (n + COLS - 1) // COLS は整数だけで書く切り上げ
    rows = (len(ranking) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 3.2 * rows))
    # axes は2次元配列 flatten で1次元にして ranking と zip できるようにする
    flat = list(axes.flatten())
    for ax, (s, c) in zip(flat, ranking):
        # 1つの科目のグラフに 4寮のヒストグラムを重ねる
        for h, values in house_scores(df, c).items():
            # alpha=0.5: 半透明にして重なりが見えるようにする
            # density=True: 面積を1にそろえる 寮ごとに人数が違っても
            #               分布の形を比べられる
            ax.hist(values, bins=BINS, alpha=0.5, density=True,
                    color=HOUSE_COLORS[h], label=h)
        ax.set_title(f"{c}  (spread {s:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
    # 科目数が奇数で余った枠は何も描かずに軸を消す
    for ax in flat[len(ranking):]:
        ax.axis("off")
    # 凡例は全グラフ共通なので 最初のグラフから取って1つだけ出す
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10)
    fig.suptitle("Score distribution per house "
                 "(left to right, top to bottom: most homogeneous first)")
    fig.tight_layout()  # グラフ同士が重ならないように余白を調整
    return fig


def main() -> None:
    df = load_scores(parse_args())   # 寮 + 科目 の表
    ranking = rank_courses(df)       # spread の小さい順に並べる
    print_ranking(ranking)           # ターミナルに順位を出す
    draw(df, ranking)                # 図を作る
    plt.show()                       # 画面に表示する


# 直接実行したときだけ main() を呼ぶ (import されたときは呼ばない)
if __name__ == "__main__":
    main()
