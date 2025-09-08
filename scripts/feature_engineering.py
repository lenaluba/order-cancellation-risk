"""Feature engineering pipeline for order cancellation risk."""

from pathlib import Path
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features for modelling.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered features.
    """
    # TODO: implement feature engineering steps
    return df
