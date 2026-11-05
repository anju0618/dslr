"""
位置
- ft_percentile: パーセント点
- ft_quartiles: 四分位数
"""

from collections.abc import Sequence

from ._core import ft_sort


def ft_percentile(x: Sequence[float], p: float) -> float:
    """
    return p パーセント点 (線形補間。pandas と同じ)
    x は昇順に並べたもの
    """
    n = len(x)
    if n == 0:
        raise ValueError("ft_percentile: empty sequence")
    if p < 0 or p > 100:
        raise ValueError("ft_percentile: p must be between 0 and 100")

    pos = (n - 1) * p / 100
    lo = int(pos)
    if lo + 1 >= n:
        return x[lo]
    return x[lo] + (pos - lo) * (x[lo + 1] - x[lo])


def ft_quartiles(x: Sequence[float]) -> tuple[float, float, float]:
    """
    return (Q1, Q2, Q3) = (P25, P50, P75)
    """
    y = ft_sort(x)
    return (ft_percentile(y, 25), ft_percentile(y, 50),
            ft_percentile(y, 75))
