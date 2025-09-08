# Order Cancellation Risk

Predict failed deliveries and returns for drop-shipping using the [Online Retail II dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II).

## Dataset
The dataset is provided by the UCI Machine Learning Repository. Licensing and usage details are available on their site. This project does **not** redistribute the raw data.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
make download && make run
```

## Folder Structure
- `data/` – raw and processed data
- `notebooks/` – exploratory analysis and modelling
- `scripts/` – data ingestion, feature engineering, modelling scripts
- `outputs/` – generated metrics, plots, SHAP values
- `docs/` – project documentation
- `tests/` – unit tests

See `Makefile` for common tasks.

## Tips
On windows create environment as mamba env create -n orders1 -f environment.yml
