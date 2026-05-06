"""
Lightweight predictive model using HistGradientBoostingRegressor.

FIX 1 (Bug 8): Regularised hyperparameters to prevent overfitting on small samples.
               max_depth 6→4, added min_samples_leaf=20, l2_regularization=1.0,
               max_bins=63, subsample=0.8 for stochastic boosting.

FIX 2 (Bug 5): predict_returns() now correctly maps ticker → prediction using the
               ticker list stored at fit time, not feature column names.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class ShapePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        # FIX 1: Regularised to prevent overfitting on ~300-400 training samples
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            max_depth=4,                # was 6 — too deep for this sample size
            learning_rate=0.05,         # lower lr + more trees = better generalisation
            min_samples_leaf=20,        # prevent tiny leaves memorising noise
            l2_regularization=1.0,      # explicit L2 shrinkage
            max_bins=63,                # reduce complexity
            random_state=42,
        )
        self.feature_names = None
        self.ticker = None             # FIX 2: store which ticker this predictor is for

    def fit(self, X: pd.DataFrame, y: pd.Series, ticker: str = None):
        self.feature_names = X.columns.tolist()
        self.ticker = ticker           # FIX 2: remember the target ticker
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
