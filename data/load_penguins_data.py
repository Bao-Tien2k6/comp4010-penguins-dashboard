from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
PENGUINS_CSV = DATA_DIR / "penguins.csv"
PENGUINS_RAW_CSV = DATA_DIR / "penguins_raw.csv"

PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
PENGUINS_RAW_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins_raw.csv"
)


def load_from_package() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both datasets from the Python palmerpenguins package."""
    from palmerpenguins import load_penguins, load_penguins_raw

    return load_penguins(), load_penguins_raw()


def load_from_urls() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback to public CSV sources if the package is unavailable."""
    return pd.read_csv(PENGUINS_URL), pd.read_csv(PENGUINS_RAW_URL)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        penguins, penguins_raw = load_from_package()
        source = "palmerpenguins Python package"
    except ImportError:
        try:
            penguins, penguins_raw = load_from_urls()
            source = "public CSV URLs"
        except URLError as exc:
            raise RuntimeError(
                "Could not load Palmer Penguins data. Install `palmerpenguins` with "
                "`pip install palmerpenguins` or rerun this script with internet access."
            ) from exc

    penguins.to_csv(PENGUINS_CSV, index=False)
    penguins_raw.to_csv(PENGUINS_RAW_CSV, index=False)

    print(f"Saved penguins data from {source}:")
    print(f"- {PENGUINS_CSV}")
    print(f"- {PENGUINS_RAW_CSV}")
    print(f"- penguins shape: {penguins.shape}")
    print(f"- penguins_raw shape: {penguins_raw.shape}")


if __name__ == "__main__":
    main()
