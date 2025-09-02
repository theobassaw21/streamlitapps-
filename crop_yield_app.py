import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# Page Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Ghana Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
)

_GLOBAL_CSS = """
/* Base styling for a friendly, clean look */
:root {
  --brand-1: #2E7D32; /* deep green */
  --brand-2: #66BB6A; /* light green */
  --brand-3: #F9FBE7; /* pale background */
  --text-1: #1B1B1B;
}

.main {
  background: linear-gradient(180deg, var(--brand-3) 0%, #FFFFFF 60%);
}

/* Title */
section[data-testid="stHeader"] div[role="img"] + div h1 {
  font-weight: 800;
}

/* Metric card */
.metric-card {
  border-radius: 14px;
  padding: 18px 20px;
  border: 1px solid rgba(0,0,0,0.06);
  background: #FFFFFF;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.metric-title {
  color: #2E7D32;
  font-weight: 700;
  font-size: 0.95rem;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 1.55rem;
  font-weight: 800;
}

.subtle {
  color: #4B5563;
  font-size: 0.9rem;
}

/* Emphasized result box */
.result-box {
  border-left: 6px solid var(--brand-2);
  padding: 14px 16px;
  background: #F1F8E9;
  border-radius: 8px;
}
"""

st.markdown(f"<style>{_GLOBAL_CSS}</style>", unsafe_allow_html=True)


# -----------------------------
# Domain Data
# -----------------------------
REGIONS: List[str] = [
    "Greater Accra", "Ashanti", "Northern", "Volta", "Western",
    "Eastern", "Central", "Bono", "Upper East", "Upper West",
]

CROPS: List[str] = ["Maize", "Yam", "Cocoa"]


@dataclass
class DataSpecs:
    num_rows: int = 800
    random_seed: int = 42


def _compute_base_yield(crop: str) -> float:
    if crop == "Maize":
        return 2.5
    if crop == "Yam":
        return 3.0
    if crop == "Cocoa":
        return 1.8
    return 2.0


def _region_adjustment(region: str) -> float:
    mapping = {
        "Northern": 0.3,
        "Ashanti": 0.15,
        "Volta": 0.1,
        "Upper East": -0.1,
        "Upper West": -0.05,
        "Western": 0.05,
        "Greater Accra": -0.05,
        "Eastern": 0.05,
        "Central": 0.0,
        "Bono": 0.1,
    }
    return mapping.get(region, 0.0)


@st.cache_data(show_spinner=False)
def generate_mock_dataset(specs: DataSpecs) -> pd.DataFrame:
    random.seed(specs.random_seed)
    np.random.seed(specs.random_seed)

    region_choices = np.random.choice(REGIONS, size=specs.num_rows)
    crop_choices = np.random.choice(CROPS, size=specs.num_rows, p=[0.5, 0.3, 0.2])

    # Simulate climate variations per region
    region_to_rain_mu = {
        "Northern": 950,
        "Ashanti": 1200,
        "Volta": 1100,
        "Upper East": 800,
        "Upper West": 850,
        "Western": 1500,
        "Greater Accra": 900,
        "Eastern": 1200,
        "Central": 1250,
        "Bono": 1150,
    }

    rainfall_values = []
    temperature_values = []
    soil_values = []
    yields = []
    pest_labels = []

    for region, crop in zip(region_choices, crop_choices):
        rain_mu = region_to_rain_mu.get(region, 1100)
        rainfall = float(np.clip(np.random.normal(loc=rain_mu, scale=180), 500, 2000))

        # Slightly warmer north on average
        temp_mu = 29.5 if region in {"Northern", "Upper East", "Upper West"} else 28.0
        temperature = float(np.clip(np.random.normal(loc=temp_mu, scale=2.5), 20, 40))

        soil_quality = int(np.clip(np.round(np.random.normal(loc=6.0, scale=1.8)), 1, 10))

        base = _compute_base_yield(crop)
        reg_adj = _region_adjustment(region)

        # Simple agronomic-inspired function: optimal rainfall ~ 1200mm, optimal temp ~ 28C
        rain_effect = 0.003 * (rainfall - 1200) - 0.0000015 * (rainfall - 1200) ** 2
        temp_effect = -0.2 * abs(temperature - 28)
        soil_effect = 0.35 * (soil_quality - 5)
        noise = np.random.normal(0, 0.35)

        y = base + reg_adj + rain_effect + temp_effect + soil_effect + noise
        y = max(0.3, float(y))

        # Pest & disease risk generation (synthetic)
        # Humid regions and higher-than-optimal temps tend to increase risk; better soils reduce it.
        rain_scaled = (rainfall - 1100.0) / 300.0
        temp_scaled = (temperature - 28.0) / 3.0
        soil_scaled = (soil_quality - 5.0) / 2.0
        region_bias = {
            "Western": 0.30,
            "Central": 0.25,
            "Ashanti": 0.20,
            "Eastern": 0.15,
            "Volta": 0.15,
            "Greater Accra": 0.10,
            "Bono": 0.10,
            "Northern": 0.05,
            "Upper East": 0.00,
            "Upper West": 0.00,
        }.get(region, 0.05)
        crop_bias = {"Cocoa": 0.15, "Yam": 0.10, "Maize": 0.05}.get(crop, 0.05)
        risk_score = -0.5 + 0.9 * rain_scaled + 0.6 * max(temp_scaled, 0.0) - 0.4 * soil_scaled + region_bias + crop_bias + float(np.random.normal(0, 0.35))
        risk_prob = 1.0 / (1.0 + math.exp(-risk_score))
        pest_label = 1 if risk_prob > 0.5 else 0

        rainfall_values.append(rainfall)
        temperature_values.append(temperature)
        soil_values.append(soil_quality)
        yields.append(y)
        pest_labels.append(pest_label)

    data = pd.DataFrame(
        {
            "region": region_choices,
            "crop": crop_choices,
            "rainfall": rainfall_values,
            "temperature": temperature_values,
            "soil_quality": soil_values,
            "yield": yields,
            "pest_disease": pest_labels,
        }
    )
    return data


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame, random_state: int = 42) -> Tuple[Pipeline, float, pd.DataFrame, pd.DataFrame]:
    X = data[["region", "crop", "rainfall", "temperature", "soil_quality"]]
    y = data["yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    categorical_features = ["region", "crop"]
    numeric_features = ["rainfall", "temperature", "soil_quality"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("rf", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = float(math.sqrt(mean_squared_error(y_test, preds)))

    return pipeline, rmse, X_train.copy(), X_test.copy()


def render_header() -> None:
    left, right = st.columns([0.80, 0.20], vertical_alignment="center")
    with left:
        st.title("🌾 Ghana Crop Yield Predictor")
        st.markdown(
            "Empower decisions with data-driven yield predictions from weather and soil conditions."
        )
    with right:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-title">Model</div>
              <div class="metric-value">Random Forest</div>
              <div class="subtle">On synthetic training data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(rmse_value: float) -> None:
    st.sidebar.header("Model Evaluation")
    st.sidebar.metric(label="RMSE (tons/ha)", value=f"{rmse_value:.2f}")
    with st.sidebar.expander("About this demo"):
        st.write(
            "This MVP uses a synthetic dataset to illustrate the user interface and modeling flow."
        )
        st.caption(
            "You can later plug in real historical datasets to improve accuracy."
        )


@st.cache_resource(show_spinner=False)
def train_pest_model(data: pd.DataFrame, random_state: int = 42) -> Tuple[Pipeline, float]:
    X = data[["region", "crop", "rainfall", "temperature", "soil_quality"]]
    y = data["pest_disease"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    categorical_features = ["region", "crop"]
    numeric_features = ["rainfall", "temperature", "soil_quality"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipeline = Pipeline(steps=[("pre", preprocessor), ("rf", clf)])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    return pipeline, auc


def render_inputs() -> Tuple[str, str, float, float, int]:
    st.subheader("Set Conditions")
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        selected_region = st.selectbox(
            "Region",
            REGIONS,
            index=2,  # Northern by default
            help="Select the Ghana region of interest.",
        )

    with c2:
        selected_crop = st.selectbox(
            "Crop type",
            CROPS,
            index=0,
            help="Choose the crop that you want to predict yield for.",
        )

    with c3:
        soil_quality = st.slider(
            "Soil quality index",
            min_value=1,
            max_value=10,
            value=6,
            help="Higher is better. Based on a composite soil fertility score.",
        )

    r1, r2 = st.columns([1, 1])
    with r1:
        rainfall_mm = st.slider(
            "Rainfall (mm)",
            min_value=500,
            max_value=2000,
            value=1100,
            step=10,
            help="Annual rainfall in millimeters.",
        )
    with r2:
        temperature_c = st.slider(
            "Temperature (°C)",
            min_value=20,
            max_value=40,
            value=28,
            step=1,
            help="Average growing season temperature.",
        )

    return selected_region, selected_crop, float(rainfall_mm), float(temperature_c), int(soil_quality)


def predict_yield(model: Pipeline, inputs: Tuple[str, str, float, float, int]) -> float:
    region, crop, rainfall, temperature, soil_quality = inputs
    df = pd.DataFrame(
        {
            "region": [region],
            "crop": [crop],
            "rainfall": [rainfall],
            "temperature": [temperature],
            "soil_quality": [soil_quality],
        }
    )
    pred = float(model.predict(df)[0])
    return max(0.0, pred)


def render_prediction(region: str, crop: str, predicted: float) -> None:
    st.markdown(
        f"""
        <div class="result-box">
            <div class="metric-title">Prediction</div>
            <div class="metric-value">{predicted:.2f} tons/hectare</div>
            <div class="subtle">🌾 Predicted yield for <b>{crop}</b> in <b>{region}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scatter(data: pd.DataFrame, user_point: Tuple[float, float]) -> None:
    st.subheader("Rainfall vs. Yield")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(data["rainfall"], data["yield"], alpha=0.35, edgecolor="none", label="Historical (synthetic)")
    ax.scatter([user_point[0]], [user_point[1]], color="#2E7D32", s=120, label="Your conditions")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Yield (tons/ha)")
    ax.set_title("Relationship between rainfall and yield")
    ax.legend(frameon=True)
    st.pyplot(fig, use_container_width=True)


def predict_pest_risk(model: Pipeline, inputs: Tuple[str, str, float, float, int]) -> Tuple[float, int]:
    region, crop, rainfall, temperature, soil_quality = inputs
    df = pd.DataFrame(
        {
            "region": [region],
            "crop": [crop],
            "rainfall": [rainfall],
            "temperature": [temperature],
            "soil_quality": [soil_quality],
        }
    )
    proba = float(model.predict_proba(df)[0][1])
    label = 1 if proba >= 0.5 else 0
    return proba, label


def render_pest_result(region: str, crop: str, risk_prob: float, label: int) -> None:
    risk_pct = risk_prob * 100.0
    severity = "High" if risk_prob >= 0.7 else ("Moderate" if risk_prob >= 0.4 else "Low")
    st.markdown(
        f"""
        <div class=\"result-box\">
            <div class=\"metric-title\">Pest & Disease Risk</div>
            <div class=\"metric-value\">{risk_pct:.0f}% ({severity})</div>
            <div class=\"subtle\">🪲 Estimated risk for <b>{crop}</b> in <b>{region}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(min(100, max(0, round(risk_pct)))))
    with st.expander("Recommended actions"):
        if risk_prob >= 0.6:
            st.write(
                "- Use resistant varieties\n"
                "- Implement IPM: scouting, pheromone traps, biological controls\n"
                "- Improve drainage and field sanitation\n"
                "- Rotate crops and remove infected residue\n"
                "- Targeted, label-compliant pesticide application if needed"
            )
        elif risk_prob >= 0.4:
            st.write(
                "- Increase field monitoring frequency\n"
                "- Maintain balanced fertilization and irrigation\n"
                "- Spot-treat hotspots; prune and dispose safely"
            )
        else:
            st.write(
                "- Maintain routine scouting and hygiene\n"
                "- Keep records and monitor weather-driven risk triggers"
            )


def main() -> None:
    render_header()

    specs = DataSpecs(num_rows=900, random_seed=42)
    data = generate_mock_dataset(specs)

    with st.spinner("Training models..."):
        model, rmse, X_train, X_test = train_model(data)
        pest_model, pest_auc = train_pest_model(data)

    render_sidebar(rmse)

    tabs = st.tabs(["Yield Prediction", "Pest & Disease"])

    with tabs[0]:
        region, crop, rainfall, temperature, soil_quality = render_inputs()
        action_col1, action_col2 = st.columns([1, 6])
        with action_col1:
            predict_clicked = st.button("Predict yield", type="primary")
        if predict_clicked:
            predicted = predict_yield(
                model, (region, crop, rainfall, temperature, soil_quality)
            )
            render_prediction(region, crop, predicted)
            render_scatter(data, user_point=(rainfall, predicted))
        else:
            st.info("Set parameters and click Predict to see the result and chart.")

    with tabs[1]:
        st.subheader("Pest & Disease Risk")
        st.caption(f"Classifier ROC-AUC: {pest_auc:.2f}")
        pregion, pcrop, prain, ptemp, psoil = render_inputs()
        pcol1, pcol2 = st.columns([1, 6])
        with pcol1:
            risk_clicked = st.button("Estimate risk", type="primary")
        if risk_clicked:
            prob, label = predict_pest_risk(
                pest_model, (pregion, pcrop, prain, ptemp, psoil)
            )
            render_pest_result(pregion, pcrop, prob, label)
        else:
            st.info("Set conditions and click Estimate risk.")

    with st.expander("Preview training data"):
        st.dataframe(data.head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

