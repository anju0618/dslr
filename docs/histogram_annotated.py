# 解説用コピー: src/histogram.py にコメントを足したもの。コードは同一。
#
# このスクリプトの目的:
#   課題の問い「どの科目で、4つの寮の点数分布が似ているか」に答える。
#   科目ごとに、4つの寮のヒストグラムを重ねて描く。
#   さらに、寮の違いの大きさを数値(spread)にして、小さい順に並べる。
#
# 全体の流れ (main):
#   parse_args -> collect_scores -> rank_courses -> print_ranking -> draw
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from load import load
from stats42 import ft_dropna, ft_mean, ft_std

# 寮ごとの色 (ホグワーツの寮カラー)。
# 辞書のキーの順番が、そのまま「寮の順番」として全体で使われる。
HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}
BINS = 20  # ヒストグラムの棒の数
COLS = 2   # 図を何列に並べるか

# 型エイリアス (長い型に名前を付けただけ。動作には影響しない)
# Scores : {科目: [Gryffindorの点リスト, Hufflepuffの点リスト, ...]}
#          リストの順番は HOUSE_COLORS の順番と同じ
# Ranking: [(spread, 科目), ...]
Scores = dict[str, list[list[float]]]
Ranking = list[tuple[float, str]]


def parse_args() -> tuple[str, str | None]:
    """Return (dataset path, save file or None). Exit on bad usage."""
    # コマンドライン引数の処理。
    # 使い方: histogram.py dataset.csv [--save out.png]
    args = sys.argv[1:]  # 先頭はスクリプト名なので除く
    save = None
    if "--save" in args:
        i = args.index("--save")
        # --save が最後にあって、ファイル名が無い場合はエラー
        if i + 1 >= len(args):
            print("Error: --save needs a file name", file=sys.stderr)
            sys.exit(1)
        save = args[i + 1]
        # "--save" とファイル名の2つを引数リストから取り除く。
        # 残りがデータセットのパスだけになる。
        del args[i:i + 2]
    # 残りがちょうど1個(データセットのパス)でなければ使い方を表示して終了
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} dataset.csv [--save out.png]",
              file=sys.stderr)
        sys.exit(1)
    return args[0], save


def collect_scores(path: str) -> Scores:
    """Return {course: [scores of each house]} (NaN removed)."""
    # CSVを読み、寮が書かれていない行は捨てる
    df = load(path).dropna(subset=["Hogwarts House"])
    # 数値の列だけを科目として使う。Index は科目ではないので除く
    courses = [c for c in df.select_dtypes(include="number").columns
               if c != "Index"]
    # 辞書内包表記。科目 c ごとに、寮 h ごとの点数リストを作る。
    #   df.loc[行の条件, 列] で「寮が h の生徒の、科目 c の点」を取り出し、
    #   ft_dropna で NaN (欠損) を取り除く。
    # 内側のリストの順番は HOUSE_COLORS の順番。
    return {
        c: [ft_dropna(df.loc[df["Hogwarts House"] == h, c].tolist())
            for h in HOUSE_COLORS]
        for c in courses
    }


def spread(groups: list[list[float]]) -> float:
    """Spread of the house means relative to the overall std.

    Formula:
        std(house means) / std(all values)
        small -> the four houses look alike
    """
    # 「寮ごとの平均が、どれだけ散らばっているか」を数値にする。
    # 全体のばらつきで割るのは、科目ごとの点数の単位の違いを消すため
    # (Arithmancy は5万点台、Care of Magical Creatures は±1程度、など)。
    #
    # 値が小さい: 4寮の平均がほぼ同じ -> 寮を見分けられない科目
    # 値が大きい: 寮ごとに平均が違う   -> 寮を見分けられる科目
    means = [ft_mean(g) for g in groups]            # 寮ごとの平均 (4個)
    values = [v for g in groups for v in g]         # 全寮の点を1つのリストに
    return ft_std(means) / ft_std(values)


def rank_courses(scores: Scores) -> Ranking:
    """Return [(spread, course)] sorted from most homogeneous."""
    # (spread, 科目名) のタプルのリストを作り、sorted で小さい順に並べる。
    # タプルは先頭の要素(spread)から比べられるので、key の指定は不要。
    return sorted((spread(g), c) for c, g in scores.items())


def print_ranking(ranking: Ranking) -> None:
    print("spread (small = homogeneous)")
    for s, c in ranking:
        print(f"  {s:.3f}  {c}")


def draw(scores: Scores, ranking: Ranking) -> Figure:
    """Draw one histogram per course, overlaying the four houses."""
    # 科目数を COLS 列で並べるのに必要な行数 (切り上げ)。
    # (n + COLS - 1) // COLS は、整数だけで書く切り上げの定石。
    rows = (len(ranking) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 3.2 * rows))
    # axes は2次元配列。flatten で1次元にして、ranking と zip できるようにする
    flat = list(axes.flatten())
    for ax, (s, c) in zip(flat, ranking):
        # 1つの科目のグラフに、4寮のヒストグラムを重ねる
        for h, values in zip(HOUSE_COLORS, scores[c]):
            # alpha=0.5: 半透明にして、重なりが見えるようにする
            # density=True: 面積を1にそろえる。寮ごとに人数が違っても
            #               分布の形を比べられる
            ax.hist(values, bins=BINS, alpha=0.5, density=True,
                    color=HOUSE_COLORS[h], label=h)
        ax.set_title(f"{c}  (spread {s:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
    # 科目数が奇数で余ったグラフ枠は、何も描かずに軸を消す
    for ax in flat[len(ranking):]:
        ax.axis("off")
    # 凡例は全グラフ共通なので、最初のグラフから取って1つだけ出す
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10)
    fig.suptitle("Score distribution per house "
                 "(left to right, top to bottom: most homogeneous first)")
    fig.tight_layout()  # グラフ同士が重ならないように余白を調整
    return fig


def show_or_save(fig: Figure, save: str | None) -> None:
    # --save が指定されていればファイルに保存、なければ画面に表示
    if save:
        fig.savefig(save, dpi=100)
        print(f"saved: {save}")
    else:
        plt.show()


def main() -> None:
    path, save = parse_args()
    scores = collect_scores(path)    # データを科目×寮のリストにまとめる
    ranking = rank_courses(scores)   # spread の小さい順に並べる
    print_ranking(ranking)           # ターミナルに順位を出す
    show_or_save(draw(scores, ranking), save)  # 図を描いて表示/保存


# このファイルを直接実行したときだけ main() を呼ぶ (import されたときは呼ばない)
if __name__ == "__main__":
    main()
