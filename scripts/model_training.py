"""Model training utilities for cancellation risk."""

from pathlib import Path
import pandas as pd


def train_model(df: pd.DataFrame, model_path: Path) -> None:
    """Train model and save to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with engineered features and target.
    model_path : Path
        Destination to save trained model.
    """
    # TODO: implement training logic
    pass
