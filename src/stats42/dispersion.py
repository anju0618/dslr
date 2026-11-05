"""Dispersion."""

from collections.abc import Sequence

from ._core import ft_max, ft_min, ft_sqrt
from .central import ft_mean
from .position import ft_quartiles


def ft_var(x: Sequence[float], ddof: int = 1) -> float:
    """Return the variance.

    Formula:
        s² = Σ(xᵢ − x̄)² / (n − ddof)

    Args:
        x: values without NaN.
        ddof: 0 population, 1 sample.

    Returns:
        Variance.
    """
    m = ft_mean(x)
    ssd = 0.0
    n = 0

    for v in x:
        ssd += (v - m) ** 2
        n += 1
    if n <= ddof:
        raise ValueError("ft_var: not enough values for the given ddof")
    return float((ssd / (n - ddof)))


def ft_std(x: Sequence[float], ddof: int = 1) -> float:
    """Return the standard deviation.

    Formula:
        s = √( Σ(xᵢ − x̄)² / (n − ddof) )

    Args:
        x: values without NaN.
        ddof: 0 population, 1 sample.

    Returns:
        Standard deviation.
    """
    return ft_sqrt(ft_var(x, ddof))


def ft_range(x: Sequence[float]) -> float:
    """Return max − min.

    Formula:
        range = max − min

    Args:
        x: values without NaN.

    Returns:
        Range.
    """
    return ft_max(x) - ft_min(x)


def ft_iqr(x: Sequence[float]) -> float:
    """Return the interquartile range.

    Formula:
        IQR = Q3 − Q1

    Args:
        x: values without NaN.

    Returns:
        IQR.
    """
    q1, _, q3 = ft_quartiles(x)
    return q3 - q1
