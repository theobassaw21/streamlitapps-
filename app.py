import hashlib
from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
	title="Smart Agri-Advisor",
	page_icon="🌱",
	layout="centered",
)


# ---------- Styles ----------
CUSTOM_CSS = """
<style>
/* Layout spacing */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Header banner */
.header-banner {
  background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 60%, #f3e9dd 100%);
  border-radius: 16px;
  padding: 24px 20px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  margin-bottom: 12px;
}
.header-title { font-size: 2rem; margin: 0; color: #2e7d32; }
.header-subtitle { font-size: 1.05rem; color: #4f5b50; margin-top: 4px; }

/* Cards */
.card { background: #ffffff; border-radius: 16px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
.card + .card { margin-top: 12px; }

/* Big button */
div.stButton > button {
  background-color: #2e7d32;
  color: #ffffff;
  border-radius: 12px;
  padding: 14px 20px;
  font-size: 1.1rem;
  width: 100%;
  border: 0;
  box-shadow: 0 4px 10px rgba(46,125,50,0.25);
}
div.stButton > button:hover { background-color: #1b5e20; }
div.stButton > button:focus { outline: none; box-shadow: 0 0 0 3px rgba(46,125,50,0.2); }

/* Typography */
.section-title { font-weight: 700; color: #2e3b2e; margin-bottom: 8px; }
.advisory-text { font-size: 1.05rem; line-height: 1.5; color: #2e3b2e; }

/* Responsive tweaks */
@media (max-width: 640px) {
  .header-title { font-size: 1.55rem; }
  .header-subtitle { font-size: 0.95rem; }
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Header ----------
st.markdown(
	"""
	<div class="header-banner">
	  <h1 class="header-title">🌱 Smart Agri-Advisor</h1>
	  <p class="header-subtitle">“Helping Farmers Make Smarter Decisions.”</p>
	</div>
	""",
	unsafe_allow_html=True,
)


# ---------- Inputs ----------
crops = ["Maize", "Cassava", "Rice"]
regions = ["Accra", "Kumasi", "Tamale"]

left_col, right_col = st.columns(2)
with left_col:
	crop = st.selectbox("🌾 Crop", options=crops, index=0)
with right_col:
	region = st.selectbox("📍 Region/Market", options=regions, index=0)

get_forecast = st.button("Get Forecast 📈", use_container_width=True)


# ---------- Data + Forecast Utilities ----------
def _seed_from(crop_name: str, region_name: str) -> int:
	key = f"{crop_name}-{region_name}".encode()
	return int(hashlib.sha256(key).hexdigest(), 16) % (2**32 - 1)


@st.cache_data(show_spinner=False)
def generate_historical_prices(crop_name: str, region_name: str, weeks: int = 16) -> pd.DataFrame:
	seed_value = _seed_from(crop_name, region_name)
	rng = np.random.default_rng(seed_value)
	base_price_map = {"Maize": 80.0, "Cassava": 50.0, "Rice": 120.0}
	base_price = base_price_map.get(crop_name, 80.0)

	trend_per_week = rng.normal(0.0, 0.8)
	seasonal = np.sin(np.linspace(0, 3 * np.pi, weeks)) * rng.uniform(0.0, 3.0)
	noise = rng.normal(0.0, 2.0, weeks)
	prices: list[float] = []
	price_level = base_price + rng.uniform(-5.0, 5.0)
	for i in range(weeks):
		price_level = price_level + trend_per_week + seasonal[i] * 0.1 + noise[i] * 0.2
		prices.append(max(base_price * 0.5, float(price_level)))

	end_date = pd.Timestamp.today().normalize()
	dates = [end_date - pd.Timedelta(weeks=(weeks - i)) for i in range(weeks)]
	data = pd.DataFrame({"Date": dates, "Price": prices})
	data["Status"] = "Past"
	return data


def forecast_next_weeks(past_prices: pd.Series, weeks_ahead: int = 3) -> np.ndarray:
	indices = np.arange(len(past_prices))
	# Simple linear trend forecast
	coeffs = np.polyfit(indices, past_prices.to_numpy(), 1)
	trend_fn = np.poly1d(coeffs)
	future_indices = np.arange(len(past_prices), len(past_prices) + weeks_ahead)
	pred = trend_fn(future_indices)
	# Ensure non-negative and reasonable smoothing
	pred = np.maximum(pred, 0.0)
	return pred


def build_chart(past_df: pd.DataFrame, forecast_df: pd.DataFrame) -> alt.Chart:
	past = past_df.copy()
	forecast = forecast_df.copy()
	base = alt.Chart().encode(
		x=alt.X("Date:T", title="Date"),
		y=alt.Y("Price:Q", title="Price"),
		tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Price:Q", title="Price")],
	)
	past_line = base.mark_line(color="#2e7d32", strokeWidth=3).encode(detail=alt.value("Past")).transform_calculate(Date="datum.Date", Price="datum.Price").transform_filter(alt.datum.Status == "Past").properties()
	forecast_line = base.mark_line(color="#8d6e63", strokeDash=[6, 4], strokeWidth=3).encode(detail=alt.value("Forecast")).transform_calculate(Date="datum.Date", Price="datum.Price").transform_filter(alt.datum.Status == "Forecast")
	forecast_points = base.mark_point(color="#8d6e63", filled=True, size=60).transform_filter(alt.datum.Status == "Forecast")
	combined = alt.layer(past_line, forecast_line, forecast_points, data=pd.concat([past, forecast], ignore_index=True)).resolve_scale(color="independent").properties(height=360)
	return combined


# ---------- Outputs ----------
if get_forecast:
	past_df = generate_historical_prices(crop, region, weeks=16)
	weeks_ahead = 3  # forecast horizon (2–4 weeks range)
	last_date = past_df["Date"].max()
	pred_values = forecast_next_weeks(past_df["Price"], weeks_ahead=weeks_ahead)
	forecast_dates = [last_date + timedelta(weeks=i + 1) for i in range(weeks_ahead)]
	forecast_df = pd.DataFrame({"Date": forecast_dates, "Price": pred_values, "Status": "Forecast"})

	# Chart card
	st.markdown('<div class="card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">📈 Price Forecast</div>', unsafe_allow_html=True)
	chart = build_chart(past_df, forecast_df)
	st.altair_chart(chart, use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)

	# Advisory card
	last_price = float(past_df["Price"].iloc[-1])
	avg_future = float(np.mean(pred_values))
	change_pct = 0.0 if last_price == 0 else (avg_future - last_price) / last_price * 100.0

	if change_pct > 2.0:
		message = f"{crop} prices in {region} may rise in the next weeks. It could be better to wait before selling."
	elif change_pct < -2.0:
		message = f"{crop} prices in {region} may fall soon. Consider selling earlier to avoid lower prices."
	else:
		message = f"{crop} prices in {region} look stable. You can sell when it is convenient."

	st.markdown('<div class="card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">📝 Advisory</div>', unsafe_allow_html=True)
	st.markdown(f'<div class="advisory-text">{message} This is only a guide. Check local market news.</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)
else:
	# Initial friendly nudge
	st.markdown('<div class="card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">📈 Price Forecast</div>', unsafe_allow_html=True)
	st.info("Pick a crop and a market, then tap ‘Get Forecast’.", icon="ℹ️")
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">📝 Advisory</div>', unsafe_allow_html=True)
	st.info("Your simple advice will appear here after the forecast.")
	st.markdown('</div>', unsafe_allow_html=True)

