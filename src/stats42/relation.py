"""
2変数の関係
- ft_corr: ピアソンの相関係数
"""

from collections.abc import Sequence

from ._core import ft_sqrt
from .central import ft_mean


def ft_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """
    return r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Σ(xᵢ − x̄)² · Σ(yᵢ − ȳ)²)
    """
    if len(x) != len(y):
        raise ValueError("ft_corr: x and y must have the same length")
    mx = ft_mean(x)
    my = ft_mean(y)
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for a, b in zip(x, y):
        sxy += (a - mx) * (b - my)
        sxx += (a - mx) * (a - mx)
        syy += (b - my) * (b - my)
    if sxx == 0 or syy == 0:
        raise ValueError("ft_corr: variance is 0")
    return sxy / ft_sqrt(sxx * syy)
