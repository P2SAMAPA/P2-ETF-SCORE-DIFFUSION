"""
Fetch and prepare data from Hugging Face dataset.
"""
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config


def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_INPUT_FILE} from {config.HF_INPUT_DATASET}...")
    file_path = hf_hub_download(
        repo_id=config.HF_INPUT_DATASET,
        filename=config.HF_INPUT_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
    )
    df = pd.read_parquet(file_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame index dtype:", df.index.dtype)

    if pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.sort_index()
        return compute_returns(df)

    if pd.api.types.is_numeric_dtype(df.index):
        sample_val = df.index[0] if len(df) > 0 else 0
        if sample_val > 1e12:
            unit = "ns"
        elif sample_val > 1e10:
            unit = "ms"
        elif sample_val > 1e9:
            unit = "s"
        else:
            unit = None
        if unit is not None:
            df.index = pd.to_datetime(df.index, unit=unit)
            df = df.sort_index()
            return compute_returns(df)

    possible_time_cols = ["__index_level_0__", "date", "Date", "timestamp", "time", "index"]
    time_col = next((c for c in possible_time_cols if c in df.columns), None)

    if time_col is not None:
        if pd.api.types.is_numeric_dtype(df[time_col]):
            sample_val = df[time_col].iloc[0]
            unit = "ns" if sample_val > 1e12 else "ms" if sample_val > 1e10 else "s" if sample_val > 1e9 else None
            df["date"] = pd.to_datetime(df[time_col], unit=unit) if unit else pd.to_datetime(df[time_col])
        else:
            df["date"] = pd.to_datetime(df[time_col])
        df = df.set_index("date")
        if time_col != "date":
            df = df.drop(columns=[time_col])
        df = df.sort_index()
        return compute_returns(df)

    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col])
            if converted.notna().all():
                df["date"] = converted
                df = df.set_index("date").drop(columns=[col]).sort_index()
                return compute_returns(df)
        except Exception:
            continue

    raise KeyError("Unable to locate date information.")


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for all price columns."""
    price_cols = [col for col in df.columns if col not in config.MACRO_COLS]
    for col in price_cols:
        df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))
    return df


def get_universe_returns(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    if universe == "fi":
        tickers = config.FI_COMMODITY_TICKERS
    elif universe == "equity":
        tickers = config.EQUITY_TICKERS
    elif universe == "combined":
        tickers = config.COMBINED_TICKERS
    else:
        raise ValueError("universe must be 'fi', 'equity', or 'combined'")

    ret_cols = [f"{t}_ret" for t in tickers if f"{t}_ret" in df.columns]
    return df[ret_cols].dropna()
