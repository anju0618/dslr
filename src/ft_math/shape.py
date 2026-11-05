"""Shape."""

from collections.abc import Sequence


def ft_skewness(x: Sequence[float]) -> float:
    """Return the adjusted skewness.

    Formula:
        G₁ = √(n(n − 1)) / (n − 2) · m₃ / m₂^(3/2)

    Args:
        x: values without NaN (n >= 3).

    Returns:
        Skewness.
    """
    raise NotImplementedError


def ft_kurtosis(x: Sequence[float]) -> float:
    """Return the adjusted excess kurtosis.

    Formula:
        G₂ = (n − 1) / ((n − 2)(n − 3)) · ((n + 1)·g₂ + 6)
        g₂ = m₄ / m₂² − 3

    Args:
        x: values without NaN (n >= 4).

    Returns:
        Excess kurtosis.
    """
    raise NotImplementedError
