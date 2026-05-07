"""
Main training script for Score Diffusion engine.

FIXES:
  Bug 3 — Ticker index mismatch: trainer iterated over config tickers
           but indexed into expected_returns using position, which broke
           silently whenever any ticker was missing from the data.
           Fixed by iterating over etf_names (the actual data columns)
           and building a dict keyed by name, not by config-list position.

  Bug 6 — Mean-of-generative-samples is not a return prediction:
           The diffusion model generates samples from the historical
           return distribution conditioned on macro. Taking mean()*252
           gives the macro-conditioned historical mean — near zero for
           all assets after StandardScaling, producing near-random ranking.

           Fix: use a relative-strength composite score that combines:
             (a) mean trajectory return (macro-conditioned level)
             (b) upside probability: P(trajectory > 0) — skewness signal
             (c) Sharpe of trajectories: mean/std — reward consistency
             (d) momentum: recent 63-day realised return (independent signal)
           
           These four components are z-scored and combined with weights
           that emphasise forward-looking signals (upside prob + momentum)
           over the near-zero generative mean.
"""

import json
import pandas as pd
import numpy as np
import torch

import config
import data_manager
from diffusion_model import DiffusionPredictor
import push_results


def _composite_score(traj_orig: np.ndarray, recent_returns: np.ndarray) -> np.ndarray:
    """
    FIX Bug 6: build a multi-factor score instead of raw mean()*252.

    traj_orig:      (num_traj, n_assets)  — trajectories in log-return space
    recent_returns: (n_assets,)           — trailing 63-day annualised log return

    Returns: (n_assets,) composite score (higher = stronger long candidate)
    """
    n_assets = traj_orig.shape[1]

    # Factor 1: macro-conditioned MEDIAN trajectory (annualised)
    # Use median — robust to outlier trajectories even after clamping
    mean_ret = np.median(traj_orig, axis=0) * 252

    # Factor 2: upside probability P(traj > 0) — rewards positive skew
    upside_prob = (traj_orig > 0).mean(axis=0)

    # Factor 3: trajectory Sharpe (mean / std of trajectories)
    traj_std   = traj_orig.std(axis=0) + 1e-8
    traj_sharpe = mean_ret / (traj_std * np.sqrt(252))

    # Factor 4: recent momentum (trailing 63-day realised return)
    momentum = recent_returns  # already annualised by caller

    # Z-score each factor across assets for comparability
    def zscore(v):
        s = v.std()
        return (v - v.mean()) / (s + 1e-8)

    z_mean    = zscore(mean_ret)
    z_upside  = zscore(upside_prob)
    z_sharpe  = zscore(traj_sharpe)
    z_momentum = zscore(momentum)

    # Weighted composite: emphasise upside prob and momentum
    score = (0.20 * z_mean
           + 0.35 * z_upside
           + 0.20 * z_sharpe
           + 0.25 * z_momentum)
    return score


def run_score_diffusion():
    print(f"=== P2-ETF-SCORE-DIFFUSION Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]

    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}
    top_picks   = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        X_ret, X_cond, scaler_ret, scaler_cond, etf_names = \
            data_manager.build_training_data(returns, macro)

        n_assets = len(etf_names)

        predictor = DiffusionPredictor(
            data_dim=n_assets,
            cond_dim=X_cond.shape[1],
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_steps=config.DIFFUSION_STEPS,
            noise_schedule=config.NOISE_SCHEDULE,
            lr=config.LEARNING_RATE,
            wd=config.WEIGHT_DECAY,
            seed=config.RANDOM_SEED
        )

        print(f"  Training diffusion model on {len(X_ret)} samples ({n_assets} assets)...")
        predictor.fit(X_ret, X_cond, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        # Sample trajectories conditioned on latest macro
        latest_cond = torch.tensor(X_cond[-1:], dtype=torch.float32)
        # FIX EXPLODING RETURNS: use 256 trajectories (was NUM_TRAJECTORIES which
        # could be 64 — too few to get a stable mean from N(0,1) samples)
        traj    = predictor.sample_trajectories(latest_cond, num_traj=256)
        traj_np = traj.cpu().numpy()   # (256, n_assets)

        # FIX: clamp in SCALED space to ±3σ before inverse-transforming.
        # Without this, outlier trajectories from an undertrained model produce
        # values with magnitude >> 1 in scaled space. At scaler.scale_ ≈ 0.01
        # (daily vol), even a scaled value of 5 → 5*0.01*252 = 1260% annualised.
        traj_np_clamped = np.clip(traj_np, -3.0, 3.0)

        # Inverse-transform to original log-return scale
        traj_orig = scaler_ret.inverse_transform(traj_np_clamped)  # (256, n_assets)

        # FIX: use MEDIAN not mean — robust to the few remaining outlier trajectories
        # after clamping. Median of 256 N(0,1)/scale samples gives a stable
        # daily return estimate with std ≈ daily_vol/sqrt(256)*1.25 ≈ 0.08% daily.
        median_daily = np.median(traj_orig, axis=0)   # (n_assets,)
        std_daily    = traj_orig.std(axis=0)          # (n_assets,)

        # FIX Bug 6: trailing 63-day momentum (annualised log return per asset)
        lookback = min(63, len(returns))
        recent_ret_ann = returns.iloc[-lookback:].mean().values * 252  # (n_assets,)

        # Composite score (uses median_daily * 252 internally)
        scores = _composite_score(traj_orig, recent_ret_ann)

        # FIX Bug 3: iterate over etf_names (actual data columns), not config tickers
        universe_results = {}
        for i, ticker in enumerate(etf_names):
            mean_ret_ann = float(median_daily[i] * 252)
            traj_std_ann = float(std_daily[i] * np.sqrt(252))
            universe_results[ticker] = {
                "ticker":           ticker,
                "composite_score":  float(scores[i]),
                # expected_return: dashboard reads this key
                "expected_return":  mean_ret_ann,
                "mean_return_ann":  mean_ret_ann,
                "upside_prob":      float((traj_orig[:, i] > 0).mean()),
                "trajectory_std":   traj_std_ann,
                "momentum_ann":     float(recent_ret_ann[i]),
            }

        all_results[universe_name] = universe_results

        # Rank by composite score
        sorted_tickers = sorted(
            universe_results.items(),
            key=lambda x: x[1]["composite_score"],
            reverse=True
        )
        top_picks[universe_name] = [
            {"ticker": t, **d} for t, d in sorted_tickers[:3]
        ]

        print(f"  Top pick: {sorted_tickers[0][0]} "
              f"(score={sorted_tickers[0][1]['composite_score']:.3f})")

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"
        },
        "daily_trading": {
            "universes":  all_results,
            "top_picks":  top_picks,
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_score_diffusion()
