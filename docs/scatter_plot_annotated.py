# 解説用コピー: src/scatter_plot.py にコメントを足したもの。コードは同一。
#
# このスクリプトの目的:
#   課題の問い「どの2つの特徴量(科目)が似ているか」に答える。
#   「似ている」= 片方の点が分かればもう片方も分かる = 相関が強い。
#   13科目の全ペア(78通り)の相関係数 r を計算し、|r| が大きい順に並べて、
#   上位のペアを散布図にする。
#
# 全体の流れ (main):
#   parse_args -> collect_table -> rank_pairs -> print_ranking -> draw
import sys
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from load import load
from stats42 import ft_corr  # ピアソンの相関係数 (自作)

HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
TOP = 6   # 散布図にする上位ペアの数
COLS = 3  # 図を何列に並べるか

# 型エイリアス (動作には影響しない)
# Table  : {科目: [全生徒の点]}  (NaN を含む。リストの長さは全科目で同じ)
# Points : (xの値のリスト, yの値のリスト, 寮のリスト)
#          同じ位置の要素が同じ生徒を指す
# Ranking: [(r, 科目a, 科目b), ...]
Table = dict[str, list[float]]
Points = tuple[list[float], list[float], list[str]]
Ranking = list[tuple[float, str, str]]


def parse_args() -> tuple[str, str | None]:
    """Return (dataset path, save file or None). Exit on bad usage."""
    # histogram.py と同じ。使い方: scatter_plot.py dataset.csv [--save out.png]
    args = sys.argv[1:]
    save = None
    if "--save" in args:
        i = args.index("--save")
        if i + 1 >= len(args):
            print("Error: --save needs a file name", file=sys.stderr)
            sys.exit(1)
        save = args[i + 1]
        del args[i:i + 2]  # "--save" とファイル名を取り除く
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} dataset.csv [--save out.png]",
              file=sys.stderr)
        sys.exit(1)
    return args[0], save


def collect_table(path: str) -> tuple[list[str], Table]:
    """Return (house of each student, {course: score of each student}).

    NaN is kept here so that every list has the same length.
    """
    # histogram.py との違い:
    #   histogram は寮ごとに分けて NaN を捨てていた。
    #   ここでは「同じ生徒の2科目」を組にする必要があるので、
    #   NaN を捨てず、全科目のリストの長さ・順番をそろえたままにする。
    #   (NaN を捨てると、位置がずれて別の生徒同士を組にしてしまう)
    df = load(path).dropna(subset=["Hogwarts House"])
    courses = [c for c in df.select_dtypes(include="number").columns
               if c != "Index"]
    houses = df["Hogwarts House"].tolist()  # 各生徒の寮 (点の色分けに使う)
    return houses, {c: df[c].tolist() for c in courses}


def pair_points(houses: list[str], table: Table, a: str, b: str) -> Points:
    """Return (xs, ys, houses) of the students who have both scores."""
    # 科目 a と b の、両方に点がある生徒だけを取り出す。
    xs, ys, hs = [], [], []
    # zip で、同じ位置 (= 同じ生徒) の a の点、b の点、寮を同時に取り出す
    for x, y, h in zip(table[a], table[b], houses):
        # NaN は自分自身と等しくない (nan == nan は False)。
        # なので x == x は「x が NaN でない」という判定になる。
        if x == x and y == y:
            xs.append(x)
            ys.append(y)
            hs.append(h)
    return xs, ys, hs


def rank_pairs(houses: list[str], table: Table) -> Ranking:
    """Return [(r, course a, course b)] sorted from the strongest |r|."""
    ranking = []
    # combinations(table, 2): 科目名から2つ選ぶ全組み合わせ。
    # 13科目なら 13*12/2 = 78 通り。(a,b) と (b,a) は同じ扱いで1回だけ。
    for a, b in combinations(table, 2):
        xs, ys, _ = pair_points(houses, table, a, b)  # 寮はここでは使わない
        ranking.append((ft_corr(xs, ys), a, b))
    # key=lambda t: -abs(t[0]) : |r| の大きい順。
    # abs を使うのは、r = -1 (逆向きの完全な直線) も「似ている」に入れるため。
    return sorted(ranking, key=lambda t: -abs(t[0]))


def print_ranking(ranking: Ranking) -> None:
    print("correlation (|r| close to 1 = similar)")
    for r, a, b in ranking[:TOP]:  # 上位 TOP 個だけ表示
        print(f"  {r:+.3f}  {a} / {b}")  # :+ は符号を必ず付ける指定


def draw(houses: list[str], table: Table, ranking: Ranking) -> Figure:
    """Draw one scatter plot per pair, colored by house."""
    top = ranking[:TOP]
    rows = (len(top) + COLS - 1) // COLS  # 切り上げ
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 4.5 * rows))
    flat = list(axes.flatten())
    for ax, (r, a, b) in zip(flat, top):
        xs, ys, hs = pair_points(houses, table, a, b)
        # 寮ごとに色を変えて、4回に分けて点を打つ
        for h, color in HOUSE_COLORS.items():
            # 寮 h の生徒の点だけを取り出す (内包表記で条件に合うものを選ぶ)
            ax.scatter([x for x, k in zip(xs, hs) if k == h],
                       [y for y, k in zip(ys, hs) if k == h],
                       s=8, alpha=0.5, color=color, label=h)
        ax.set_xlabel(a, fontsize=8)  # 横軸: 科目a
        ax.set_ylabel(b, fontsize=8)  # 縦軸: 科目b
        ax.set_title(f"r = {r:+.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat[len(top):]:  # 余った枠は消す
        ax.axis("off")
    # 凡例は全グラフ共通なので、最初のグラフから取って1つだけ出す
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10)
    fig.suptitle("Most correlated pairs of courses "
                 "(left to right, top to bottom: strongest first)")
    fig.tight_layout()
    return fig


def show_or_save(fig: Figure, save: str | None) -> None:
    if save:
        fig.savefig(save, dpi=100)
        print(f"saved: {save}")
    else:
        plt.show()


def main() -> None:
    path, save = parse_args()
    houses, table = collect_table(path)       # 全科目の点と、各生徒の寮
    ranking = rank_pairs(houses, table)       # 全78ペアを |r| の大きい順に
    print_ranking(ranking)
    show_or_save(draw(houses, table, ranking), save)


if __name__ == "__main__":
    main()
