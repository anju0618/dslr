"""
ばらつき
- ft_var: 分散
- ft_std: 標準偏差
- ft_range: 範囲
- ft_iqr: 四分位範囲
"""

from collections.abc import Sequence

from ._core import ft_max, ft_min, ft_sqrt
from .central import ft_mean
from .position import ft_quartiles


def ft_var(x: Sequence[float], ddof: int = 1) -> float:
    """
    return s² = Σ(xᵢ − x̄)² / (n − ddof)
    ddof=1 は標本分散 ddof=0 は母分散
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
    """
    return s = √( Σ(xᵢ − x̄)² / (n − ddof) )
    """
    return ft_sqrt(ft_var(x, ddof))


def ft_range(x: Sequence[float]) -> float:
    """
    return max − min
    """
    return ft_max(x) - ft_min(x)


def ft_iqr(x: Sequence[float]) -> float:
    """
    return IQR = Q3 − Q1
    """
    q1, _, q3 = ft_quartiles(x)
    return q3 - q1
