"""Position."""

from collections.abc import Sequence

from ._core import ft_sort


def ft_percentile(x: Sequence[float], p: float) -> float:
    """Return the p-th percentile (linear interpolation).

    Formula:
        pos = (n − 1)·p/100
        P = y_lo + frac·(y_hi − y_lo)

    Note:
        There are several percentile definitions. This is the linear one
        (numpy / pandas default, Excel PERCENTILE.INC), so it matches
        pandas describe(). e.g. [1, 2, 3, 4] at 25% -> 1.75
        (nearest rank gives 1, pos = (n + 1)·p/100 gives 1.25).

    Args:
        x: values sorted ascending, without NaN.
        p: 0 to 100.

    Returns:
        Percentile.
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
    """Return (Q1, Q2, Q3).

    Formula:
        Q1 = P25, Q2 = P50, Q3 = P75

    Args:
        x: values without NaN (any order).

    Returns:
        Quartiles.
    """
    y = ft_sort(x)
    return (ft_percentile(y, 25), ft_percentile(y, 50),
            ft_percentile(y, 75))
