import hashlib
from datetime import timedelta, datetime

import altair as alt
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI


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
.chat-container { max-height: 420px; overflow-y: auto; padding: 16px; scroll-behavior: smooth; }
.chat-message { margin: 8px 0; display: flex; }
.chat-bubble { padding: 12px 14px; border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); max-width: 92%; }
.assistant .chat-bubble { background: #f1f8e9; color: #2e3b2e; border-top-left-radius: 6px; }
.user { justify-content: flex-end; }
.user .chat-bubble { background: #2e7d32; color: #ffffff; border-top-right-radius: 6px; }
.quick-actions { border-top: 1px solid #e8ebe6; padding: 10px 16px 14px 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.qa-btn { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 999px; padding: 8px 12px; font-size: 0.95rem; }
.chat-meta { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
.typing { display: inline-block; }
.typing .dot { height: 6px; width: 6px; margin: 0 2px; background: #9ca3af; border-radius: 50%; display: inline-block; animation: blink 1.4s infinite both; }
.typing .dot:nth-child(2) { animation-delay: .2s; }
.typing .dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0% { opacity: .2; } 20% { opacity: 1; } 100% { opacity: .2; } }

/* Design tokens */
:root {
  --primary: #2e7d32;
  --primary-dark: #1b5e20;
  --secondary: #8d6e63;
  --bg: #FAFAF8;
  --card: #ffffff;
}

/* Hero banner */
.hero {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  margin-bottom: 14px;
}
.hero::before {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(120deg, rgba(46,125,50,0.95), rgba(129,199,132,0.85));
}
.hero::after {
  content: "";
  position: absolute; inset: 0;
  background-image: url('https://images.unsplash.com/photo-1464226184884-fa280b87c399?q=80&w=1600&auto=format&fit=crop');
  background-size: cover; background-position: center; opacity: 0.20;
}
.hero-content { position: relative; padding: 34px 22px; text-align: center; color: #ffffff; }
.hero-title { font-size: 2.2rem; font-weight: 800; margin: 0; }
.hero-sub { font-size: 1.15rem; margin-top: 6px; opacity: 0.95; }
.hero-bar { height: 6px; background: linear-gradient(90deg, #66bb6a, #43a047, #8d6e63); }

/* Input card */
.input-card { background: linear-gradient(180deg, #ffffff, #f7fbf7); }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Header ----------
st.markdown(
	"""
	<div class="hero">
	  <div class="hero-content">
	    <h1 class="hero-title">🌱 Smart Agri-Advisor</h1>
	    <p class="hero-sub">Your AI-powered guide for better farming decisions</p>
	  </div>
	  <div class="hero-bar"></div>
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

# Input selection card (centered)
st.markdown('<div class="card input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title" style="text-align:center;">What would you like to know?</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#4f5b50;">Select your crop and market location for personalized advice</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
	crop = st.selectbox("🌾 Crop", options=crops, index=0)
with c2:
	region = st.selectbox("📍 Market/Region", options=regions, index=0)

get_forecast = st.button("🚀 Get Smart Advice", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


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
	combined = pd.concat([past_df.assign(Status="Historical"), forecast_df.assign(Status="Predicted")], ignore_index=True)
	color_scale = alt.Scale(domain=["Historical", "Predicted"], range=["#3b82f6", "#10b981"])
	tooltip = [
		alt.Tooltip("Date:T", title="Date"),
		alt.Tooltip("Price:Q", title="Price (GHS)", format=".2f"),
		alt.Tooltip("Status:N", title="Type"),
	]
	base = alt.Chart(combined).encode(
		x=alt.X("Date:T", title="Date"),
		y=alt.Y("Price:Q", title="Price (GHS)", stack=None),
		color=alt.Color("Status:N", scale=color_scale, legend=alt.Legend(title="")),
		tooltip=tooltip,
	)
	area = base.mark_area(opacity=0.3).encode()
	line = base.mark_line(size=3).encode()
	points = base.mark_point(size=40, filled=True).encode()
	chart = alt.layer(area, line, points).properties(height=340)
	return chart


# ---------- AI Chat Helper (OpenAI) ----------
@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI | None:
	try:
		api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
		if not api_key:
			api_key = os.environ.get("OPENAI_API_KEY")
		if not api_key:
			return None
		return OpenAI(api_key=api_key)
	except Exception:
		return None


def ai_reply(context: dict, user_message: str) -> str | None:
	try:
		client = get_openai_client()
		if client is None:
			return None
	except Exception:
		return None
	sys_prompt = (
		"You are Smart Agri-Advisor, a friendly farming assistant."
		" Speak simply, 2-4 short sentences, reference crop and market,"
		" and offer one concrete next step. Avoid technical ML terms."
	)
	ctx = (
		f"Crop: {context.get('crop')} | Market: {context.get('region')} | "
		f"Current Price: {context.get('last_price'):.2f} | Forecast Change (%): {context.get('change_pct'):+.1f}"
	)
	try:
		resp = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": sys_prompt},
				{"role": "user", "content": f"Context: {ctx}\nQuestion: {user_message}"},
			],
			max_tokens=180,
			temperature=0.4,
		)
		return resp.choices[0].message.content.strip()
	except Exception:
		return None


# ---------- Chat Advisor + Forecast ----------
if "chat" not in st.session_state:
	st.session_state.chat = []  # list of (role, text)

# Keep latest forecast in session so free-form chat can reuse
if "latest_past_df" not in st.session_state:
	st.session_state.latest_past_df = None
if "latest_forecast_df" not in st.session_state:
	st.session_state.latest_forecast_df = None
if "latest_metrics" not in st.session_state:
	st.session_state.latest_metrics = None

# One-time reset to remove any old auto-seeded messages
if st.session_state.get("app_version") != "no_autoseed_v1":
	st.session_state.chat = []
	st.session_state.latest_past_df = None
	st.session_state.latest_forecast_df = None
	st.session_state.latest_metrics = None
	st.session_state.app_version = "no_autoseed_v1"

def add_chat(role: str, text: str) -> None:
	st.session_state.chat.append((role, text))

def extract_crop_region_from_text(text: str, default_crop: str, default_region: str) -> tuple[str, str]:
	lc = text.lower()
	crop_names = ["maize", "rice", "cassava", "cocoa", "yam", "plantain"]
	region_names = ["accra", "kumasi", "tamale", "cape coast", "ho", "sunyani", "takoradi", "bolgatanga"]
	chosen_crop = next((c for c in crop_names if c in lc), default_crop)
	if "cape coast" in lc:
		chosen_region = "Cape Coast"
	else:
		chosen_region = next((r.title() for r in region_names if r in lc), default_region)
	return chosen_crop.capitalize(), chosen_region

def set_latest(past_df: pd.DataFrame, forecast_df: pd.DataFrame, last_price: float, change_pct: float, best_date: pd.Timestamp, best_price: float) -> None:
	st.session_state.latest_past_df = past_df
	st.session_state.latest_forecast_df = forecast_df
	st.session_state.latest_metrics = {
		"last_price": last_price,
		"change_pct": change_pct,
		"best_date": best_date,
		"best_price": best_price,
	}

def extract_crop_region_from_text(text: str, default_crop: str, default_region: str) -> tuple[str, str]:
	lc = text.lower()
	crop_names = ["maize", "rice", "cassava", "cocoa", "yam", "plantain"]
	region_names = ["accra", "kumasi", "tamale", "cape coast", "ho", "sunyani", "takoradi", "bolgatanga"]
	chosen_crop = next((c for c in crop_names if c in lc), clean_name(default_crop))
	if "cape coast" in lc:
		chosen_region = "cape coast"
	else:
		chosen_region = next((r for r in region_names if r in lc), clean_name(default_region))
	return chosen_crop.capitalize(), chosen_region.title()

# No default chat message; advice appears only after user action

# Chat panel
st.markdown('<div class="card chat-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤖 Your AI Advisor <span style="font-size:0.9rem;color:#43a047;">● online</span></div>', unsafe_allow_html=True)
# Clear chat button
if st.button("🗑️ Clear chat", key="clear_chat", help="Clear this conversation"):
	st.session_state.chat = []
	st.session_state.latest_past_df = None
	st.session_state.latest_forecast_df = None
	st.session_state.latest_metrics = None
st.markdown('<div class="chat-container" id="chatbox">', unsafe_allow_html=True)
for role, text in st.session_state.chat:
	css_cls = "assistant" if role == "assistant" else "user"
	stamp = datetime.now().strftime("%I:%M %p").lstrip("0")
	avatar = '🤖 ' if css_cls == 'assistant' else ''
	st.markdown(f'<div class="chat-message {css_cls}"><div class="chat-bubble">{avatar}{text}<div class="chat-meta">{stamp}</div></div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Free-form chat input for follow-ups
user_query = st.chat_input("Ask me anything about farming…")
if user_query:
	add_chat("user", user_query)
	# Try to extract crop/region from message; fallback to selected ones
	default_crop = clean_name(crop)
	default_region = clean_name(region)
	q_crop, q_region = extract_crop_region_from_text(user_query, default_crop, default_region)
	# Forecast for this query
	past_df = generate_historical_prices_days(q_crop, q_region, days=14)
	last_date = past_df["Date"].max()
	try:
		pred_values = forecast_with_ml_steps(past_df["Price"], steps_ahead=10)
	except Exception:
		pred_values = forecast_next_steps(past_df["Price"], steps_ahead=10)
	forecast_dates = [last_date + timedelta(days=i + 1) for i in range(len(pred_values))]
	forecast_df = pd.DataFrame({"Date": forecast_dates, "Price": pred_values, "Status": "Forecast"})
	last_price = float(past_df["Price"].iloc[-1])
	avg_future = float(np.mean(pred_values))
	change_pct = 0.0 if last_price == 0 else (avg_future - last_price) / last_price * 100.0
	best_idx = int(np.argmax(pred_values)) if change_pct >= 1.5 else (0 if change_pct <= -1.5 else min(3, len(forecast_dates) - 1))
	best_date = forecast_dates[best_idx]
	best_price = float(pred_values[best_idx])
	direction_text = "rise" if change_pct > 1.5 else ("fall" if change_pct < -1.5 else "stay stable")
	# Prefer AI response if available
	resp = None
	context = {"crop": q_crop, "region": q_region, "last_price": last_price, "change_pct": change_pct}
	ai_text = ai_reply(context, user_query)
	if ai_text:
		resp = ai_text
	else:
		resp = (
			f"For {q_crop} in {q_region}, prices may {direction_text} about {abs(change_pct):.0f}% over the next 10 days. "
			+ (
				"If storage is safe, consider waiting until " + best_date.strftime("%b %d") + f" (around {best_price:.2f}). "
				if change_pct > 1.5 else
				"Selling sooner may protect your income. " if change_pct < -1.5 else
				"Prices look steady—sell when convenient. "
			)
			+ "Want me to compare with another market like Accra or Tamale?"
		)
	# Typing indicator
	st.markdown('<div class="chat-message assistant"><div class="chat-bubble"><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div></div>', unsafe_allow_html=True)
	time.sleep(0.5)
	add_chat("assistant", resp)
	# Save latest for chart + download
	st.session_state.latest_past_df = past_df
	st.session_state.latest_forecast_df = forecast_df
	st.session_state.latest_metrics = {"last_price": last_price, "change_pct": change_pct}

# Render latest forecast (cards + chart) if available
if st.session_state.latest_past_df is not None and st.session_state.latest_forecast_df is not None:
	st.markdown('<div class="card">', unsafe_allow_html=True)
	title_crop = clean_name(crop)
	title_region = clean_name(region)
	st.markdown(f'<div class="section-title">📈 Price Forecast for {title_crop} in {title_region}</div>', unsafe_allow_html=True)
	m = st.session_state.latest_metrics or {}
	m1, m2, m3 = st.columns(3)
	with m1:
		if "last_price" in m:
			st.metric("Current Price", f"{m['last_price']:.2f}")
	with m2:
		if "change_pct" in m:
			st.metric("Forecast Change", f"{abs(m['change_pct']):.0f}%", delta=f"{m['change_pct']:+.1f}%")
	with m3:
		if "change_pct" in m:
			action = "Wait" if m['change_pct'] > 1.5 else ("Sell Now" if m['change_pct'] < -1.5 else "Flexible")
			st.metric("Recommended Action", action)
	chart = build_chart(st.session_state.latest_past_df, st.session_state.latest_forecast_df)
	st.altair_chart(chart, use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)

	# Sidebar download
	with st.sidebar:
		combined = pd.concat([st.session_state.latest_past_df, st.session_state.latest_forecast_df], ignore_index=True)
		csv_bytes = combined.to_csv(index=False).encode("utf-8")
		st.download_button(
			"⬇️ Download data (CSV)",
			data=csv_bytes,
			file_name=f"smart_agri_prices_forecast.csv",
			mime="text/csv",
			use_container_width=True,
		)

# When farmer taps Get Advice
if get_forecast:
	# Clear any previous auto-messages to avoid clutter
	st.session_state.chat = []
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

	# Generate assistant advice without adding a user-simulated message
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
	# Prefer AI response for button-based advice too
	context = {"crop": _crop, "region": _region, "last_price": last_price, "change_pct": change_pct}
	ai_text = ai_reply(context, f"Give advice for {_crop} in {_region} for the next 10 days.")
	bot_msg = ai_text if ai_text else (
		f"Based on the latest price trends, {_crop} in {_region} could {direction_text} by about {abs(change_pct):.0f}% over the next 10 days. "
		+ advice_tail +
		"Would you like me to also check Tamale or Accra for comparison?"
	)
	# Simulate typing indicator
	st.markdown('<div class="chat-message assistant"><div class="chat-bubble"><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div></div>', unsafe_allow_html=True)
	# Small delay to feel conversational
	time.sleep(0.5)
	add_chat("assistant", bot_msg)

	# Store latest forecast for rendering below and in downloads
	st.session_state.latest_past_df = past_df
	st.session_state.latest_forecast_df = forecast_df
	st.session_state.latest_metrics = {"last_price": last_price, "change_pct": change_pct}

