"""Functions for loading the CICIDS2017 Parquet dataset."""
from pathlib import Path
import pandas as pd


def load_parquet_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and combine all Parquet files in a directory."""
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in: {data_dir.resolve()}")

    frames = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("The dataset must contain a 'Label' target column.")

    return df
