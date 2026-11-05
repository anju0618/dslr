"""
前処理
- ft_standardize: 標準化 (z スコア)
- ft_normalize: min-max 正規化
"""

from collections.abc import Sequence


def ft_standardize(
        x: Sequence[float],
        mean: float,
        std: float
        ) -> list[float]:
    """
    return zᵢ = (xᵢ − mean) / std
    mean, std は学習データのもの
    """
    if std == 0:
        raise ValueError("ft_standardize: std must not be 0")
    return [(v - mean) / std for v in x]


def ft_normalize(x: Sequence[float], lo: float, hi: float) -> list[float]:
    """
    return x'ᵢ = (xᵢ − lo) / (hi − lo)
    lo と hi は学習データの最小値と最大値
    """
    if hi == lo:
        raise ValueError("ft_normalize: hi must differ from lo")
    return [(v - lo) / (hi - lo) for v in x]
