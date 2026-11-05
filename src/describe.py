import sys

from load import load
from stats42 import ft_describe
from stats42.summary import BASIC, EXTRA


def format_table(
        result: dict[str, dict[str, float]],
        rows: tuple[str, ...]
        ) -> str:
    """Format the statistics as a table (one column per feature)."""
    label_w = max(len(r) for r in rows)
    lines = []
    cells = {
        name: [f"{result[name][r]:.6f}" for r in rows] for name in result
    }
    widths = {
        name: max(len(name), *(len(c) for c in cells[name]))
        for name in result
    }
    head = " " * label_w + "".join(
        f"  {name:>{widths[name]}}" for name in result
    )
    lines.append(head)
    for i, r in enumerate(rows):
        line = f"{r:<{label_w}}" + "".join(
            f"  {cells[name][i]:>{widths[name]}}" for name in result
        )
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--extra"]
    extra = len(args) != len(sys.argv) - 1
    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} dataset.csv [--extra]", file=sys.stderr)
        sys.exit(1)

    df = load(args[0])
    names = [c for c in df.select_dtypes(include="number").columns
             if c != "Index"]
    if not names:
        print("Error: no numerical feature", file=sys.stderr)
        sys.exit(1)

    result = ft_describe({c: df[c].tolist() for c in names}, extra=extra)
    rows = BASIC + EXTRA if extra else BASIC
    print(format_table(result, rows))


if __name__ == "__main__":
    main()
