"""Basic building blocks."""

from collections.abc import Sequence


def ft_dropna(x: Sequence[float]) -> list[float]:
    """Drop NaN values.

    Formula:
        keep v where v == v

    Args:
        x: values, may contain NaN.

    Returns:
        New list without NaN.
    """
    res: list[float] = []
    for i in x:
        if i == i:
            res.append(i)

    return res


def ft_count(x: Sequence[float]) -> int:
    """Count the items.

    Formula:
        n = Σ 1

    Args:
        x: values without NaN.

    Returns:
        n.
    """
    count: int = 0
    for i in x:
        count += 1
    return count


def ft_sum(x: Sequence[float]) -> float:
    """Sum the values.

    Formula:
        S = Σ xᵢ

    Args:
        x: values without NaN.

    Returns:
        Sum (0.0 if empty).
    """
    res: float = 0.0
    for i in x:
        res += i
    return res


def ft_min(x: Sequence[float]) -> float:
    """Return the minimum.

    Formula:
        min = xₘ, xₘ ≤ xᵢ for all i

    Args:
        x: values without NaN.

    Returns:
        Minimum.
    """
    if len(x) == 0:
        raise ValueError("ft_min: empty sequence")
    res = x[0]
    for i in x:
        if res > i:
            res = i

    return res


def ft_max(x: Sequence[float]) -> float:
    """Return the maximum.

    Formula:
        max = xₘ, xₘ ≥ xᵢ for all i

    Args:
        x: values without NaN.

    Returns:
        Maximum.
    """
    if len(x) == 0:
        raise ValueError("ft_max: empty sequence")
    res = x[0]
    for i in x:
        if res < i:
            res = i

    return res


def ft_sqrt(x: float) -> float:
    """Return the square root.

    Formula:
        yₖ₊₁ = (yₖ + x / yₖ) / 2  (Newton)

    Args:
        x: number >= 0.

    Returns:
        √x.
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
    """Merge two ascending lists into one."""
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
    """Return a new ascending list (merge sort, O(n log n)).

    Formula:
        y₁ ≤ y₂ ≤ ... ≤ yₙ

    Args:
        x: values without NaN.

    Returns:
        Sorted list.
    """
    if len(x) <= 1:
        return list(x)
    mid = len(x) // 2
    return _merge(ft_sort(x[:mid]), ft_sort(x[mid:]))
