"""Population Stability Index (PSI) utilities."""

import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate the PSI between two distributions.

    Parameters
    ----------
    expected : pandas.Series
        Expected or training distribution.
    actual : pandas.Series
        Actual or scoring distribution.
    buckets : int, default=10
        Number of quantile buckets.

    Returns
    -------
    float
        The PSI value.
    """
    # TODO: implement PSI calculation
    return 0.0
