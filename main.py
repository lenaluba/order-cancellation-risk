"""Entry point for the end-to-end pipeline."""

from pathlib import Path
import pandas as pd

from scripts.data_ingestion import download_data
from scripts.feature_engineering import engineer_features
from scripts.model_training import train_model


def main():
    """Run the full data pipeline."""
    excel_path = download_data()
    # Placeholder: read Excel and clean
    df = pd.read_excel(excel_path)
    # TODO: implement cleaning (hash CustomerID, truncate timestamps)
    features = engineer_features(df)
    # TODO: create target variable 'full_basket_return'
    train_model(features, Path('outputs/model.pkl'))


if __name__ == "__main__":
    main()
