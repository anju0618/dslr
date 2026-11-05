"""Central tendency."""

from collections.abc import Sequence


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
    raise NotImplementedError


def ft_mode(x: Sequence[float]) -> float:
    """Return the most frequent value.

    Formula:
        argmax_v Σ [xᵢ == v]

    Args:
        x: values without NaN.

    Returns:
        Mode (smallest on ties).
    """
    raise NotImplementedError
