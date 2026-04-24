"""
Configuration for P2-ETF-SCORE-DIFFUSION engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-score-diffusion-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Diffusion Parameters ---
DIFFUSION_STEPS = 100                 # Number of diffusion steps
HIDDEN_DIM = 128                      # Hidden layer size
NUM_LAYERS = 4                        # Residual layers
TIME_EMBED_DIM = 64                   # Time embedding dimension
COND_EMBED_DIM = 32                   # Condition embedding dimension
NOISE_SCHEDULE = "cosine"             # "linear" or "cosine"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 500                          # Training epochs
BATCH_SIZE = 128
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252

# --- Sampling ---
NUM_TRAJECTORIES = 64                # Number of diffusion trajectories per ETF

# --- Training Data ---
TRAIN_START = "2008-01-01"

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
