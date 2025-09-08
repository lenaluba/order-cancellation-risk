"""Tools for analyzing risk thresholds."""

import pandas as pd


def analyze_thresholds(probs: pd.Series, targets: pd.Series) -> pd.DataFrame:
    """Compute metrics across risk thresholds.

    Parameters
    ----------
    probs : pandas.Series
        Predicted probabilities.
    targets : pandas.Series
        True labels.

    Returns
    -------
    pandas.DataFrame
        Metrics per threshold.
    """
    # TODO: implement threshold analysis
    return pd.DataFrame()
