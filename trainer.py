"""
Main training script for Score Diffusion engine.
"""

import json
import pandas as pd
import numpy as np
import torch

import config
import data_manager
from diffusion_model import DiffusionPredictor
import push_results

def run_score_diffusion():
    print(f"=== P2-ETF-SCORE-DIFFUSION Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]

    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        X_ret, X_cond, scaler_ret, scaler_cond, etf_names = data_manager.build_training_data(returns, macro)

        predictor = DiffusionPredictor(
            data_dim=X_ret.shape[1],
            cond_dim=X_cond.shape[1],
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_steps=config.DIFFUSION_STEPS,
            noise_schedule=config.NOISE_SCHEDULE,
            lr=config.LEARNING_RATE,
            wd=config.WEIGHT_DECAY,
            seed=config.RANDOM_SEED
        )

        print(f"  Training diffusion model on {len(X_ret)} samples...")
        predictor.fit(X_ret, X_cond, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        # Predict using the most recent macro condition
        latest_cond = torch.tensor(X_cond[-1:], dtype=torch.float32)
        traj = predictor.sample_trajectories(latest_cond, num_traj=config.NUM_TRAJECTORIES)
        traj_np = traj.cpu().numpy()  # (num_traj, n_assets)

        # Inverse-transform to original return scale
        traj_orig = scaler_ret.inverse_transform(traj_np)  # daily log returns
        expected_returns = traj_orig.mean(axis=0) * 252      # annualized

        # Build results
        universe_results = {}
        for i, ticker in enumerate(tickers):
            universe_results[ticker] = {
                "ticker": ticker,
                "expected_return": expected_returns[i],
                "trajectory_std": float(traj_orig[:, i].std() * np.sqrt(252))
            }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["expected_return"], reverse=True)
        top_picks[universe_name] = [
            {k: v for k, v in d.items() if k != 'ticker'} | {"ticker": t}
            for t, d in sorted_tickers[:3]
        ]

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_score_diffusion()
