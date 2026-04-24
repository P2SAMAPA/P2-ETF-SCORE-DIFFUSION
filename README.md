# P2-ETF-SCORE-DIFFUSION

**Score‑Based Diffusion Model (DDPM) for ETF Return Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-SCORE-DIFFUSION/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-SCORE-DIFFUSION/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--score--diffusion--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-score-diffusion-results)

## Overview

`P2-ETF-SCORE-DIFFUSION` uses a **Denoising Diffusion Probabilistic Model (DDPM)** to learn the conditional distribution of ETF returns given macro features. Trained on 2008–2026 YTD, the model generates multiple return trajectories and ranks ETFs by their expected return across diffusion paths.

## Methodology

- **Diffusion Process**: 100‑step cosine noise schedule.
- **Score Network**: Residual MLP with time and condition embeddings.
- **Training**: 200 epochs on the full historical dataset.
- **Inference**: 64 trajectories sampled for the latest macro condition.
- **Ranking**: Top ETFs by trajectory‑mean expected return.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
