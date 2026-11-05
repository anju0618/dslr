"""Position."""

from collections.abc import Sequence


def ft_percentile(x: Sequence[float], p: float) -> float:
    """Return the p-th percentile (linear interpolation).

    Formula:
        pos = (n − 1)·p/100
        P = y_lo + frac·(y_hi − y_lo)

    Args:
        x: values without NaN.
        p: 0 to 100.

    Returns:
        Percentile.
    """
    raise NotImplementedError


def ft_quartiles(x: Sequence[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3).

    Formula:
        Q1 = P25, Q2 = P50, Q3 = P75

    Args:
        x: values without NaN.

    Returns:
        Quartiles.
    """
    raise NotImplementedError
