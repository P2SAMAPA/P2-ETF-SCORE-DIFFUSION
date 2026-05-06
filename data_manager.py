"""
Data loading and preprocessing for Score Diffusion engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare wide-format log returns."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Extract macro columns and forward-fill."""
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_training_data(returns: pd.DataFrame, macro: pd.DataFrame) -> tuple:
    """
    Returns:
        X_ret: (n_samples, n_assets) – log returns at each date
        X_cond: (n_samples, cond_dim) – macro features at each date
        scaler_ret: StandardScaler fitted on returns
        scaler_cond: StandardScaler fitted on macro
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    scaler_ret = StandardScaler()
    scaler_cond = StandardScaler()
    X_ret = scaler_ret.fit_transform(returns.values)
    X_cond = scaler_cond.fit_transform(macro.values)
    return X_ret, X_cond, scaler_ret, scaler_cond, returns.columns.tolist()
