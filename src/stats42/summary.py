"""Summary of numeric columns (describe)."""

from collections.abc import Mapping, Sequence

from ._core import ft_dropna, ft_sort
from .central import ft_mean, ft_mode
from .dispersion import ft_std, ft_var
from .position import ft_percentile
from .shape import ft_kurtosis, ft_skewness

NAN = float("nan")

BASIC = ("Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max")
EXTRA = ("Var", "Range", "IQR", "Mode", "Skew", "Kurt", "Missing")


def _describe_column(x: Sequence[float], extra: bool) -> dict[str, float]:
    """Compute the statistics of one column."""
    y = ft_sort(ft_dropna(x))
    n = len(y)
    res: dict[str, float] = {name: NAN for name in BASIC}
    res["Count"] = float(n)
    if extra:
        res.update({name: NAN for name in EXTRA})
        res["Missing"] = float(len(x) - n)
    if n == 0:
        return res

    res["Mean"] = ft_mean(y)
    res["Min"] = y[0]
    res["25%"] = ft_percentile(y, 25)
    res["50%"] = ft_percentile(y, 50)
    res["75%"] = ft_percentile(y, 75)
    res["Max"] = y[-1]
    if n >= 2:
        res["Std"] = ft_std(y)
    if extra:
        res["Range"] = y[-1] - y[0]
        res["IQR"] = res["75%"] - res["25%"]
        res["Mode"] = ft_mode(y)
        if n >= 2:
            res["Var"] = ft_var(y)
        if n >= 3 and res["Var"] > 0:
            res["Skew"] = ft_skewness(y)
        if n >= 4 and res["Var"] > 0:
            res["Kurt"] = ft_kurtosis(y)
    return res


def ft_describe(
        columns: Mapping[str, Sequence[float]],
        extra: bool = False
        ) -> dict[str, dict[str, float]]:
    """Describe numeric columns.

    Formula:
        Count, Mean, Std (ddof=1), Min, Q(25), Q(50), Q(75), Max
        extra=True adds Var, Range, IQR, Mode, Skew, Kurt, Missing

    Args:
        columns: column name -> values (NaN allowed).
        extra: add the bonus statistics.

    Returns:
        {column name: {statistic name: value}}. Statistics that cannot be
        computed (too few values) are NaN.
    """
    return {name: _describe_column(x, extra) for name, x in columns.items()}
