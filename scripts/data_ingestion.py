"""Download and extract the Online Retail II dataset."""

import hashlib
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
TARGET_PATH = Path("data/raw/online_retail_II.xlsx")
ZIP_PATH = Path("data/raw/online_retail_ii.zip")
# TODO: update with actual SHA-256 hash of the zip file
EXPECTED_SHA256 = """572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"""


def sha256sum(filename: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_data() -> Path:
    """Download and unzip dataset if not already present.

    Returns
    -------
    Path
        Path to the extracted Excel file.
    """
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TARGET_PATH.exists():
        print("Dataset already downloaded. Skipping download.")
        return TARGET_PATH

    if not ZIP_PATH.exists():
        print(f"Downloading dataset from {DATA_URL}...")
        urlretrieve(DATA_URL, ZIP_PATH)
    else:
        print("Zip file already present. Skipping download.")

    file_hash = sha256sum(ZIP_PATH)

    if EXPECTED_SHA256 != "" and file_hash != EXPECTED_SHA256:
        raise ValueError("SHA-256 mismatch: file may be corrupted")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".xlsx")]
        if not members:
            raise ValueError("No Excel file found in the zip")
        # extract first Excel file
        member = members[0]
        zf.extract(member, TARGET_PATH.parent)
        extracted_path = TARGET_PATH.parent / member
        extracted_path.rename(TARGET_PATH)
    return TARGET_PATH


if __name__ == "__main__":
    download_data()
