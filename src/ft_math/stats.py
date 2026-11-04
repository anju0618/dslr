import numpy as np


def ft_mean(x: np.ndarray) -> float:
    """
    return mean
    """
    return float(x.sum() / len(x))


def ft_std(x: np.ndarray, ddof: int = 0) -> float:
    """
    return std
    ddof=0: population std, ddof=1: sample std
    """
    m = ft_mean(x)
    return float(np.sqrt(((x - m) ** 2).sum() / (len(x) - ddof)))


def ft_standardize(
        original: np.ndarray,
        mean: float,
        std: float
        ) -> np.ndarray:
    """
    return standardization (z-score)
    """
    res = (original - mean) / std
    return res
