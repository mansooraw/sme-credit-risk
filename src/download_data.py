"""Download the raw SBA loan dataset from Kaggle.

The file SBAnational.csv (179 MB) is larger than GitHub's 100 MB file
limit, so it is not committed. This script fetches it with kagglehub,
which does not need a Kaggle API key for a public dataset, and copies it
to data/raw/ where every notebook expects it.

Usage (from the repository root):
    python src/download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

from config import KAGGLE_DATASET, RAW_DATA_DIR, RAW_DATA_PATH, RAW_FILE_NAME


def download() -> Path:
    """Download the dataset and return the path of the raw CSV file."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        print(f"Raw file already present: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    print(f"Downloading {KAGGLE_DATASET} from Kaggle ...")
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    source = next(cache_dir.rglob(RAW_FILE_NAME))
    shutil.copy2(source, RAW_DATA_PATH)
    print(f"Saved to {RAW_DATA_PATH} ({RAW_DATA_PATH.stat().st_size / 1e6:.0f} MB)")
    return RAW_DATA_PATH


if __name__ == "__main__":
    download()
