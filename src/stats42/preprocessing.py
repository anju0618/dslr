"""Preprocessing."""

from collections.abc import Sequence


def ft_standardize(
        x: Sequence[float],
        mean: float,
        std: float
        ) -> list[float]:
    """Standardize values (z-score).

    Formula:
        zᵢ = (xᵢ − mean) / std

    Args:
        x: values without NaN.
        mean: mean of the train data.
        std: std of the train data.

    Returns:
        Standardized list.
    """
    if std == 0:
        raise ValueError("ft_standardize: std must not be 0")
    return [(v - mean) / std for v in x]


def ft_normalize(x: Sequence[float], lo: float, hi: float) -> list[float]:
    """Scale values to [0, 1] (min-max).

    Formula:
        x'ᵢ = (xᵢ − lo) / (hi − lo)

    Args:
        x: values without NaN.
        lo: min of the train data.
        hi: max of the train data.

    Returns:
        Normalized list.
    """
    if hi == lo:
        raise ValueError("ft_normalize: hi must differ from lo")
    return [(v - lo) / (hi - lo) for v in x]
