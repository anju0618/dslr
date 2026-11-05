"""
代表値
- ft_mean: 平均
- ft_median: 中央値
- ft_mode: 最頻値
"""

from collections.abc import Sequence
from ._core import ft_sort


def ft_mean(x: Sequence[float]) -> float:
    """
    return x̄ = (1 / n) Σ xᵢ
    """
    total = 0.0
    n = 0
    for v in x:
        total += v
        n += 1
    if n == 0:
        raise ValueError("ft_mean: empty sequence")
    return total / n


def ft_median(x: Sequence[float]) -> float:
    """
    return 中央値 (昇順に並べた真ん中の値)
    """
    if len(x) == 0:
        raise ValueError("ft_median: empty sequence")

    x = ft_sort(x)
    y = len(x)
    if y % 2 == 0:
        return (x[y // 2] + x[y // 2 - 1]) / 2
    else:
        return x[y // 2]


def ft_mode(x: Sequence[float]) -> float:
    """
    return 最頻値 (同数なら最小の値)
    """
    if len(x) == 0:
        raise ValueError("ft_mode: empty sequence")

    counts: dict[float, int] = {}
    for v in x:
        counts[v] = counts.get(v, 0) + 1

    res = x[0]
    best = 0
    for v, c in counts.items():
        if c > best or (c == best and v < res):
            res = v
            best = c
    return res
