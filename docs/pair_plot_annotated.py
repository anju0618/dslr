# 解説用コピー: src/pair_plot.py にコメントを足したもの。コードは同一。
#
# このスクリプトの目的:
#   ロジスティック回帰に使う特徴量(科目)を選ぶために、
#   全科目の組み合わせを1枚の図(散布図行列)にして見る。
#     対角線   : その科目のヒストグラム (寮ごとに色分け)
#     それ以外 : 2科目の散布図       (寮ごとに色分け)
#   寮の色がはっきり分かれる科目が、予測に使える特徴量。
#
# 全体の流れ (main):
#   parse_args -> collect_table -> draw -> show_or_save
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from load import load
from stats42 import ft_dropna

HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
BINS = 20    # 対角線のヒストグラムの棒の数
CELL = 1.8   # 1マスの大きさ (インチ)。全体の図のサイズは CELL * 科目数

# 型エイリアス (動作には影響しない)
# Table: {科目: {寮: その寮の生徒の点のリスト}}
Table = dict[str, dict[str, list[float]]]


def parse_args() -> tuple[str, str | None]:
    """Return (dataset path, save file or None). Exit on bad usage."""
    # histogram.py、scatter_plot.py と同じ
    args = sys.argv[1:]
    save = None
    if "--save" in args:
        i = args.index("--save")
        if i + 1 >= len(args):
            print("Error: --save needs a file name", file=sys.stderr)
            sys.exit(1)
        save = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} dataset.csv [--save out.png]",
              file=sys.stderr)
        sys.exit(1)
    return args[0], save


def collect_table(path: str) -> Table:
    """Return {course: {house: scores of the house}}.

    NaN is kept so that, in a given house, every course has the same
    students at the same positions (needed to pair two courses).
    """
    # 寮ごとに分けているが、NaN は捨てない。
    # 同じ寮の中では、全科目のリストの長さと生徒の順番が同じになる。
    # 散布図で x と y を組にするとき、位置が合っている必要があるため。
    df = load(path).dropna(subset=["Hogwarts House"])
    courses = [c for c in df.select_dtypes(include="number").columns
               if c != "Index"]
    return {
        c: {h: df.loc[df["Hogwarts House"] == h, c].tolist()
            for h in HOUSE_COLORS}
        for c in courses
    }


def draw(table: Table) -> Figure:
    """Draw the scatter plot matrix.

    Diagonal: histogram of the course. Others: scatter plot of
    (column course, row course). Colored by house.
    """
    courses = list(table)
    n = len(courses)  # 科目数 (13)。図は n x n のマス目になる
    fig, axes = plt.subplots(n, n, figsize=(CELL * n, CELL * n))
    # i: 行番号 (縦軸の科目)、j: 列番号 (横軸の科目)
    for i, row in enumerate(courses):
        for j, col in enumerate(courses):
            ax = axes[i][j]
            for h, color in HOUSE_COLORS.items():
                if i == j:
                    # 対角線: 同じ科目同士なので、散布図ではなくヒストグラム。
                    # ヒストグラムは NaN を扱えないので ft_dropna で取り除く
                    ax.hist(ft_dropna(table[col][h]), bins=BINS, alpha=0.5,
                            color=color)
                else:
                    # それ以外: 横軸が列の科目、縦軸が行の科目の散布図。
                    # 散布図は NaN の点を matplotlib が自動で飛ばしてくれる。
                    # (x と y の位置を保つため、ここでは NaN を捨てない)
                    ax.scatter(table[col][h], table[row][h], s=2, alpha=0.5,
                               color=color)
            # 169マスもあるので、目盛りは消して見やすくする
            ax.set_xticks([])
            ax.set_yticks([])
            # 科目名は、左端の列(縦軸)と、下端の行(横軸)にだけ付ける
            if j == 0:
                ax.set_ylabel(row, fontsize=6, rotation=0, ha="right")
            if i == n - 1:
                ax.set_xlabel(col, fontsize=6, rotation=45, ha="right")
    # 凡例用に、寮の色の丸印を作る (全マス共通なので図全体に1つだけ)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c)
               for c in HOUSE_COLORS.values()]
    fig.legend(handles, list(HOUSE_COLORS), loc="upper right", fontsize=10)
    fig.suptitle("Pair plot (diagonal: histogram, others: scatter plot)")
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
    table = collect_table(path)       # 科目 x 寮 の点を集める
    show_or_save(draw(table), save)   # 散布図行列を描いて表示/保存


if __name__ == "__main__":
    main()
