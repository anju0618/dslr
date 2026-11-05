"""Shape."""

from collections.abc import Sequence

from ._core import ft_sqrt
from .central import ft_mean


def ft_skewness(x: Sequence[float], bias: bool = False) -> float:
    """Return the skewness.

    Formula:
        g₁ = m₃ / m₂^(3/2)                       (bias=True)
        G₁ = √(n(n − 1)) / (n − 2) · g₁          (bias=False)

    Args:
        x: values without NaN (n >= 3 if bias=False).
        bias: True for the population form g₁, False for the adjusted
            sample form G₁ (same as pandas skew()).

    Returns:
        Skewness.
    """
    n = len(x)
    if n < (1 if bias else 3):
        raise ValueError("ft_skewness: not enough values")
    m = ft_mean(x)
    m2 = 0.0
    m3 = 0.0
    for v in x:
        d = v - m
        m2 += d * d
        m3 += d * d * d
    m2 /= n
    m3 /= n
    if m2 == 0:
        raise ValueError("ft_skewness: variance is 0")
    g1 = m3 / (m2 * ft_sqrt(m2))
    if bias:
        return g1
    return ft_sqrt(n * (n - 1)) / (n - 2) * g1


def ft_kurtosis(x: Sequence[float], bias: bool = False) -> float:
    """Return the excess kurtosis.

    Formula:
        g₂ = m₄ / m₂² − 3                                (bias=True)
        G₂ = (n − 1) / ((n − 2)(n − 3)) · ((n + 1)·g₂ + 6)   (bias=False)

    Args:
        x: values without NaN (n >= 4 if bias=False).
        bias: True for the population form g₂, False for the adjusted
            sample form G₂ (same as pandas kurt()).

    Returns:
        Excess kurtosis.
    """
    n = len(x)
    if n < (1 if bias else 4):
        raise ValueError("ft_kurtosis: not enough values")
    m = ft_mean(x)
    m2 = 0.0
    m4 = 0.0
    for v in x:
        d = (v - m) * (v - m)
        m2 += d
        m4 += d * d
    m2 /= n
    m4 /= n
    if m2 == 0:
        raise ValueError("ft_kurtosis: variance is 0")
    g2 = m4 / (m2 * m2) - 3
    if bias:
        return g2
    return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
