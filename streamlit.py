"""
Streamlit Dashboard for Score Diffusion Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Score Diffusion", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-positive { color: #28a745; font-weight: 600; }
    .metric-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("score_diffusion_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def return_badge(val):
    if val >= 0:
        return f'<span class="metric-positive">+{val*100:.2f}%</span>'
    return f'<span class="metric-negative">{val*100:.2f}%</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🌊 P2Quant Score Diffusion</div>', unsafe_allow_html=True)
st.markdown('<div>DDPM – Score‑Based Generative Model for ETF Return Forecasting</div>', unsafe_allow_html=True)

with st.expander("📘 How It Works", expanded=False):
    st.markdown("""
    **Score Diffusion**: A denoising diffusion probabilistic model learns to reverse a gradual noising process.
    Trained on full 2008–2026 data, it generates multiple return trajectories conditioned on current macro variables.
    ETFs are ranked by their average expected return across trajectories.
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key, [])
        universe_data = universes.get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            ret = pick['expected_return']
            std = pick.get('trajectory_std', 0.0)
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🌊 TOP PICK (Diffusion‑Averaged Return)</div>
                <div class="hero-ticker">{ticker}</div>
                <div>Expected Return: {return_badge(ret)}</div>
                <div style="margin-top: 0.5rem;">Trajectory Std: {std*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Picks")
            rows = []
            for p in top:
                rows.append({
                    "Ticker": p['ticker'],
                    "Expected Return": f"{p['expected_return']*100:.2f}%",
                    "Trajectory Std": f"{p.get('trajectory_std', 0.0)*100:.2f}%"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, d in universe_data.items():
                all_rows.append({
                    "Ticker": t,
                    "Expected Return": f"{d['expected_return']*100:.2f}%",
                    "Trajectory Std": f"{d.get('trajectory_std', 0.0)*100:.2f}%"
                })
            df_all = pd.DataFrame(all_rows).sort_values("Expected Return", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
