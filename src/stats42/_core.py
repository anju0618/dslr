"""
基本部品
- ft_dropna: NaN の除去
- ft_count: 要素数
- ft_sum: 合計
- ft_min / ft_max: 最小値 / 最大値
- ft_sqrt: 平方根
- ft_sort: 昇順の整列
"""

from collections.abc import Sequence


def ft_dropna(x: Sequence[float]) -> list[float]:
    """
    return NaN を除いたリスト (v == v の要素だけ残す)
    """
    res: list[float] = []
    for i in x:
        if i == i:
            res.append(i)

    return res


def ft_count(x: Sequence[float]) -> int:
    """
    return 要素数 n = Σ 1
    """
    count: int = 0
    for i in x:
        count += 1
    return count


def ft_sum(x: Sequence[float]) -> float:
    """
    return 合計 S = Σ xᵢ
    """
    res: float = 0.0
    for i in x:
        res += i
    return res


def ft_min(x: Sequence[float]) -> float:
    """
    return 最小値
    """
    if len(x) == 0:
        raise ValueError("ft_min: empty sequence")
    res = x[0]
    for i in x:
        if res > i:
            res = i

    return res


def ft_max(x: Sequence[float]) -> float:
    """
    return 最大値
    """
    if len(x) == 0:
        raise ValueError("ft_max: empty sequence")
    res = x[0]
    for i in x:
        if res < i:
            res = i

    return res


def ft_sqrt(x: float) -> float:
    """
    return √x (ニュートン法: yₖ₊₁ = (yₖ + x / yₖ) / 2)
    """
    if x == 0:
        return 0.0
    elif x < 0:
        raise ValueError("ft_sqrt: arg must be positive.")

    res: float = x if x >= 1 else 1.0

    while True:
        new = (res + x / res) / 2
        if new >= res:
            return res
        res = new


def _merge(left: list[float], right: list[float]) -> list[float]:
    """
    return 昇順の2つのリストを1つの昇順リストにしたもの
    """
    res: list[float] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res


def ft_sort(x: Sequence[float]) -> list[float]:
    """
    return 昇順に並べた新しいリスト (マージソート)
    """
    if len(x) <= 1:
        return list(x)
    mid = len(x) // 2
    return _merge(ft_sort(x[:mid]), ft_sort(x[mid:]))
