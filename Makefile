.PHONY: run test clean download

run:
python main.py

test:
pytest

clean:
rm -rf outputs/*

download:
python scripts/data_ingestion.py
