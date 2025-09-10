import hashlib
from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(
	page_title="Smart Agri-Advisor",
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

/* Chat styles */
.chat-panel { padding: 0; }
.chat-container { max-height: 420px; overflow-y: auto; padding: 16px; }
.chat-message { margin: 8px 0; display: flex; }
.chat-bubble { padding: 12px 14px; border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); max-width: 92%; }
.assistant .chat-bubble { background: #f1f8e9; color: #2e3b2e; border-top-left-radius: 6px; }
.user { justify-content: flex-end; }
.user .chat-bubble { background: #2e7d32; color: #ffffff; border-top-right-radius: 6px; }
.quick-actions { border-top: 1px solid #e8ebe6; padding: 10px 16px 14px 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.qa-btn { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 999px; padding: 8px 12px; font-size: 0.95rem; }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Header ----------
st.markdown(
	"""
	<div class="header-banner">
	  <h1 class="header-title">🌱 Smart Agri-Advisor</h1>
	  <p class="header-subtitle">“Your AI-powered guide for better farming decisions.”</p>
	</div>
	""",
	unsafe_allow_html=True,
)


# ---------- Inputs (Sidebar) ----------
crops = [
	"🌽 Maize",
	"🌾 Rice",
	"🍠 Cassava",
	"🍫 Cocoa",
	"🍯 Yam",
	"🍌 Plantain",
]
regions = [
	"📍 Accra",
	"📍 Kumasi",
	"📍 Tamale",
	"📍 Cape Coast",
	"📍 Takoradi",
	"📍 Ho",
	"📍 Sunyani",
	"📍 Bolgatanga",
]

with st.sidebar:
	st.header("⚙️ Controls")
	crop = st.selectbox("🌾 Crop", options=crops, index=0)
	region = st.selectbox("📍 Market/Region", options=regions, index=0)
	get_forecast = st.button("Get Smart Advice 🧑🏾‍🌾", use_container_width=True)


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


def clean_name(label: str) -> str:
	first_space = label.find(" ")
	return label[first_space + 1 :] if first_space > 0 else label


@st.cache_data(show_spinner=False)
def generate_historical_prices_days(crop_name: str, region_name: str, days: int = 14) -> pd.DataFrame:
	seed_value = _seed_from(crop_name, region_name)
	rng = np.random.default_rng(seed_value)
	base_price_map = {
		"Maize": 80.0,
		"Cassava": 50.0,
		"Rice": 120.0,
		"Cocoa": 150.0,
		"Yam": 70.0,
		"Plantain": 60.0,
	}
	base_price = base_price_map.get(crop_name, 80.0)

	trend_per_day = rng.normal(0.0, 0.15)
	seasonal = np.sin(np.linspace(0, 2 * np.pi, days)) * rng.uniform(0.0, 1.2)
	noise = rng.normal(0.0, 1.2, days)
	prices: list[float] = []
	price_level = base_price + rng.uniform(-3.0, 3.0)
	for i in range(days):
		price_level = price_level + trend_per_day + seasonal[i] * 0.2 + noise[i] * 0.2
		prices.append(max(base_price * 0.5, float(price_level)))

	end_date = pd.Timestamp.today().normalize()
	dates = [end_date - pd.Timedelta(days=(days - i)) for i in range(days)]
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


def forecast_next_steps(past_prices: pd.Series, steps_ahead: int = 7) -> np.ndarray:
	indices = np.arange(len(past_prices))
	coeffs = np.polyfit(indices, past_prices.to_numpy(), 1)
	trend_fn = np.poly1d(coeffs)
	future_indices = np.arange(len(past_prices), len(past_prices) + steps_ahead)
	pred = trend_fn(future_indices)
	return np.maximum(pred, 0.0)


def _create_lag_features(series: np.ndarray, num_lags: int = 6) -> tuple[np.ndarray, np.ndarray, int]:
	X: list[list[float]] = []
	y: list[float] = []
	for i in range(num_lags, len(series)):
		lags = [float(series[i - j - 1]) for j in range(num_lags)]
		idx = i
		season_sin = np.sin(2 * np.pi * idx / 12.0)
		season_cos = np.cos(2 * np.pi * idx / 12.0)
		X.append(lags + [idx, season_sin, season_cos])
		y.append(float(series[i]))
	return np.array(X), np.array(y), len(series)


def forecast_with_ml(past_prices: pd.Series, weeks_ahead: int = 3) -> tuple[np.ndarray, dict]:
	series = past_prices.to_numpy().astype(float)
	if len(series) < 10:
		# Fallback to simple forecast when too little data
		return forecast_next_weeks(past_prices, weeks_ahead), {"model": "Trend", "mae": None, "r2": None}

	lag_count = min(8, max(4, len(series) // 3))
	X, y, next_index = _create_lag_features(series, num_lags=lag_count)
	if len(X) < 6:
		return forecast_next_weeks(past_prices, weeks_ahead), {"model": "Trend", "mae": None, "r2": None}

	model = RandomForestRegressor(
		n_estimators=200,
		random_state=42,
		max_depth=6,
		min_samples_leaf=1,
	)
	model.fit(X, y)

	# Simple rolling one-step backtest over last 4 points (if available)
	backtest_window = min(4, len(series) - lag_count)
	mae_vals: list[float] = []
	if backtest_window > 0:
		for t in range(len(series) - backtest_window, len(series)):
			train_series = series[:t]
			X_tr, y_tr, _ = _create_lag_features(train_series, num_lags=lag_count)
			if len(X_tr) < 3:
				continue
			m_bt = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=6)
			m_bt.fit(X_tr, y_tr)
			# Build features for time t using previous lags
			lags = [float(series[t - j - 1]) for j in range(lag_count)]
			idx = t
			feat = np.array([lags + [idx, np.sin(2 * np.pi * idx / 12.0), np.cos(2 * np.pi * idx / 12.0)]])
			pred_t = float(m_bt.predict(feat)[0])
			mae_vals.append(abs(pred_t - float(series[t])))
	mae = float(np.mean(mae_vals)) if mae_vals else None

	# Recursive multi-step forecast
	forecast: list[float] = []
	window = list(series[-lag_count:])
	for step in range(weeks_ahead):
		idx = next_index + step
		lags = [float(window[-j - 1]) for j in range(lag_count)]
		feat = np.array([[
			*lags,
			idx,
			np.sin(2 * np.pi * idx / 12.0),
			np.cos(2 * np.pi * idx / 12.0),
		]])
		pred_val = float(model.predict(feat)[0])
		pred_val = max(0.0, pred_val)
		forecast.append(pred_val)
		window.append(pred_val)
		if len(window) > lag_count:
			window.pop(0)

	return np.array(forecast), {"model": "RandomForest", "mae": mae, "r2": None}


def forecast_with_ml_steps(past_prices: pd.Series, steps_ahead: int = 7) -> np.ndarray:
	series = past_prices.to_numpy().astype(float)
	if len(series) < 10:
		return forecast_next_steps(past_prices, steps_ahead)
	lag_count = min(6, max(4, len(series) // 3))
	X, y, next_index = _create_lag_features(series, num_lags=lag_count)
	if len(X) < 4:
		return forecast_next_steps(past_prices, steps_ahead)
	model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)
	model.fit(X, y)
	forecast: list[float] = []
	window = list(series[-lag_count:])
	for step in range(steps_ahead):
		idx = next_index + step
		lags = [float(window[-j - 1]) for j in range(lag_count)]
		feat = np.array([[*lags, idx, np.sin(2 * np.pi * idx / 12.0), np.cos(2 * np.pi * idx / 12.0)]])
		pred_val = float(model.predict(feat)[0])
		pred_val = max(0.0, pred_val)
		forecast.append(pred_val)
		window.append(pred_val)
		if len(window) > lag_count:
			window.pop(0)
	return np.array(forecast)


def build_chart(past_df: pd.DataFrame, forecast_df: pd.DataFrame) -> alt.Chart:
	tooltip = [alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Price:Q", title="Price")]
	past_line = (
		alt.Chart(past_df)
		.mark_line(color="#2e7d32", strokeWidth=3)
		.encode(
			x=alt.X("Date:T", title="Date"),
			y=alt.Y("Price:Q", title="Price"),
			tooltip=tooltip,
		)
	)
	forecast_line = (
		alt.Chart(forecast_df)
		.mark_line(color="#8d6e63", strokeDash=[6, 4], strokeWidth=3)
		.encode(
			x=alt.X("Date:T", title="Date"),
			y=alt.Y("Price:Q", title="Price"),
			tooltip=tooltip,
		)
	)
	forecast_points = (
		alt.Chart(forecast_df)
		.mark_point(color="#8d6e63", filled=True, size=60)
		.encode(
			x=alt.X("Date:T", title="Date"),
			y=alt.Y("Price:Q", title="Price"),
			tooltip=tooltip,
		)
	)
	return alt.layer(past_line, forecast_line, forecast_points).properties(height=360)


# ---------- Chat Advisor + Forecast ----------
if "chat" not in st.session_state:
	st.session_state.chat = []  # list of (role, text)

def add_chat(role: str, text: str) -> None:
	st.session_state.chat.append((role, text))

# Seed a friendly greeting once
if not st.session_state.chat:
	add_chat("assistant", "Hello! Pick your crop and market, then tap ‘Get Smart Advice’. I’ll guide you.")

# Chat panel
st.markdown('<div class="card chat-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤝 Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, text in st.session_state.chat:
	css_cls = "assistant" if role == "assistant" else "user"
	st.markdown(f'<div class="chat-message {css_cls}"><div class="chat-bubble">{text}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# When farmer taps Get Advice
if get_forecast:
	# Data + forecast
	_crop = clean_name(crop)
	_region = clean_name(region)
	past_df = generate_historical_prices_days(_crop, _region, days=14)
	last_date = past_df["Date"].max()
	try:
		pred_values = forecast_with_ml_steps(past_df["Price"], steps_ahead=10)
	except Exception:
		pred_values = forecast_next_steps(past_df["Price"], steps_ahead=10)
	forecast_dates = [last_date + timedelta(days=i + 1) for i in range(len(pred_values))]
	forecast_df = pd.DataFrame({"Date": forecast_dates, "Price": pred_values, "Status": "Forecast"})

	# Simple conversation turn
	add_chat("user", f"I have {_crop} in {_region}.")
	last_price = float(past_df["Price"].iloc[-1])
	avg_future = float(np.mean(pred_values))
	change_pct = 0.0 if last_price == 0 else (avg_future - last_price) / last_price * 100.0
	direction_text = "rise" if change_pct > 1.5 else ("fall" if change_pct < -1.5 else "stay stable")
	best_idx = int(np.argmax(pred_values)) if change_pct >= 1.5 else (0 if change_pct <= -1.5 else min(3, len(forecast_dates) - 1))
	best_date = forecast_dates[best_idx]
	best_price = float(pred_values[best_idx])
	advice_tail = (
		"If you can store safely, consider waiting until " + best_date.strftime("%b %d") + f" (around {best_price:.2f}). "
		if change_pct > 1.5 else
		"Selling sooner may protect your income. " if change_pct < -1.5 else
		"Prices look steady. Sell when it suits your needs. "
	)
	bot_msg = (
		f"Based on the latest price trends, {_crop} in {_region} could {direction_text} by about {abs(change_pct):.0f}% over the next 10 days. "
		+ advice_tail +
		"Would you like me to also check Tamale or Accra for comparison?"
	)
	add_chat("assistant", bot_msg)

	# Forecast chart under chat
	st.markdown('<div class="card">', unsafe_allow_html=True)
	st.markdown('<div class="section-title">📈 Price Forecast</div>', unsafe_allow_html=True)
	chart = build_chart(past_df, forecast_df)
	st.altair_chart(chart, use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)

	# Sidebar download
	with st.sidebar:
		combined = pd.concat([past_df, forecast_df], ignore_index=True)
		csv_bytes = combined.to_csv(index=False).encode("utf-8")
		st.download_button(
			"⬇️ Download data (CSV)",
			data=csv_bytes,
			file_name=f"{crop}_{region}_prices_forecast.csv",
			mime="text/csv",
			use_container_width=True,
		)

