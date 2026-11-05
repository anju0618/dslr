"""Central tendency."""

from collections.abc import Sequence
from ._core import ft_sort


def ft_mean(x: Sequence[float]) -> float:
    """Return the arithmetic mean.

    Formula:
        x̄ = (1 / n) Σ xᵢ

    Args:
        x: values without NaN.

    Returns:
        Mean.
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
    """Return the median.

    Formula:
        middle of sorted values (P50)

    Args:
        x: values without NaN.

    Returns:
        Median.
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
    """Return the most frequent value.

    Formula:
        argmax_v Σ [xᵢ == v]

    Args:
        x: values without NaN.

    Returns:
        Mode (smallest on ties).
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
