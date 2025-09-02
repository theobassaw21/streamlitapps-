import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import torch
from torchvision import transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import os
try:
    import onnxruntime as ort
except Exception:
    ort = None
import os
try:
    import onnxruntime as ort
except Exception:
    ort = None
import os
try:
    import onnxruntime as ort
except Exception:
    ort = None


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


def _remove_unused():
    return None


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


def _placeholder_removed():
    return None


# -----------------------------
# Supply Chain Optimization
# -----------------------------
def default_supply_chain() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suppliers = pd.DataFrame(
        {
            "supplier": ["Tamale Hub", "Kumasi Hub", "Takoradi Hub"],
            "capacity_tons": [800.0, 600.0, 500.0],
        }
    )
    markets = pd.DataFrame(
        {
            "market": ["Accra", "Bolgatanga", "Ho", "Cape Coast"],
            "demand_tons": [500.0, 400.0, 300.0, 200.0],
        }
    )
    costs = pd.DataFrame(
        {
            "Accra": [55.0, 40.0, 30.0],
            "Bolgatanga": [35.0, 50.0, 70.0],
            "Ho": [60.0, 45.0, 35.0],
            "Cape Coast": [65.0, 38.0, 28.0],
        },
        index=["Tamale Hub", "Kumasi Hub", "Takoradi Hub"],
    )
    return suppliers, markets, costs


def optimize_greedy(
    suppliers: pd.DataFrame, markets: pd.DataFrame, costs: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    suppliers = suppliers.copy()
    markets = markets.copy()
    costs = costs.copy()

    if "supplier" not in suppliers.columns or "capacity_tons" not in suppliers.columns:
        raise ValueError("Suppliers must have columns: supplier, capacity_tons")
    if "market" not in markets.columns or "demand_tons" not in markets.columns:
        raise ValueError("Markets must have columns: market, demand_tons")

    # Ensure cost matrix indices/columns match names
    costs.index = costs.index.astype(str)
    costs.columns = [str(c) for c in costs.columns]
    supplier_names = [str(s) for s in suppliers["supplier"].tolist()]
    market_names = [str(m) for m in markets["market"].tolist()]

    missing_rows = [s for s in supplier_names if s not in costs.index]
    missing_cols = [m for m in market_names if m not in costs.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"Cost matrix mismatch. Missing rows: {missing_rows}, missing cols: {missing_cols}"
        )

    remaining_supply = {row["supplier"]: float(row["capacity_tons"]) for _, row in suppliers.iterrows()}
    remaining_demand = {row["market"]: float(row["demand_tons"]) for _, row in markets.iterrows()}

    # Build all edges with unit cost
    edges = []
    for s in supplier_names:
        for m in market_names:
            edges.append((float(costs.loc[s, m]), s, m))
    edges.sort(key=lambda x: x[0])

    shipments = []
    total_cost = 0.0
    for unit_cost, s, m in edges:
        if remaining_supply[s] <= 0:
            continue
        if remaining_demand[m] <= 0:
            continue
        qty = min(remaining_supply[s], remaining_demand[m])
        if qty <= 0:
            continue
        shipments.append({
            "supplier": s,
            "market": m,
            "quantity_tons": qty,
            "unit_cost": unit_cost,
            "cost": qty * unit_cost,
        })
        remaining_supply[s] -= qty
        remaining_demand[m] -= qty
        total_cost += qty * unit_cost

        # Stop early if all demand met
        if all(v <= 1e-9 for v in remaining_demand.values()):
            break

    allocation = pd.DataFrame(shipments)
    return allocation, float(total_cost)


def render_supply_chain_tab() -> None:
    st.subheader("Supply Chain Optimization")
    st.caption("Allocate shipments from hubs to markets to minimize transport cost.")

    if "sc_suppliers" not in st.session_state or "sc_markets" not in st.session_state or "sc_costs" not in st.session_state:
        s, m, c = default_supply_chain()
        st.session_state.sc_suppliers = s
        st.session_state.sc_markets = m
        st.session_state.sc_costs = c

    st.markdown("Edit suppliers (capacity), markets (demand), and the cost matrix (currency per ton):")
    colA, colB = st.columns([1, 1])
    with colA:
        st.write("Suppliers")
        st.session_state.sc_suppliers = st.data_editor(
            st.session_state.sc_suppliers,
            num_rows="dynamic",
            use_container_width=True,
        )
        st.write("Markets")
        st.session_state.sc_markets = st.data_editor(
            st.session_state.sc_markets,
            num_rows="dynamic",
            use_container_width=True,
        )
    with colB:
        st.write("Cost matrix (rows: suppliers, columns: markets)")
        st.session_state.sc_costs = st.data_editor(
            st.session_state.sc_costs,
            num_rows="dynamic",
            use_container_width=True,
        )

    run = st.button("Optimize allocation", type="primary")
    if run:
        try:
            allocation, total_cost = optimize_greedy(
                st.session_state.sc_suppliers, st.session_state.sc_markets, st.session_state.sc_costs
            )
        except Exception as e:
            st.error(f"Optimization error: {e}")
            return

        st.markdown(
            f"""
            <div class=\"metric-card\">
              <div class=\"metric-title\">Total transport cost</div>
              <div class=\"metric-value\">{total_cost:,.2f}</div>
              <div class=\"subtle\">currency units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not allocation.empty:
            st.write("Allocation plan")
            st.dataframe(allocation, use_container_width=True, hide_index=True)

            # Simple visualization: tons shipped per market
            by_market = allocation.groupby("market")["quantity_tons"].sum().reindex(
                st.session_state.sc_markets["market"].tolist()
            ).fillna(0.0)
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(by_market.index, by_market.values, color="#66BB6A")
            ax.set_ylabel("Tons shipped")
            ax.set_title("Shipments by market")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No feasible shipments generated with the current inputs.")


def main() -> None:
    render_header()

    specs = DataSpecs(num_rows=900, random_seed=42)
    data = generate_mock_dataset(specs)

    with st.spinner("Training models..."):
        model, rmse, X_train, X_test = train_model(data)

    render_sidebar(rmse)

    tabs = st.tabs(["Yield Prediction", "Pest & Disease (Images)", "Supply Chain"])

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
        st.subheader("Pest & Disease Image Classification")
        st.caption("Upload a plant/leaf image — prediction runs automatically.")

        # Lazy imports for ONNX
        import os
        try:
            import onnxruntime as ort
        except Exception:
            ort = None

        @st.cache_resource(show_spinner=False)
        def load_backbone():
            device = torch.device("cpu")
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            backbone.classifier = torch.nn.Identity()
            backbone.eval()
            backbone.to(device)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return backbone, preprocess, device

        @st.cache_resource(show_spinner=False)
        def train_demo_classifier():
            rng = np.random.default_rng(0)
            classes = np.array(["Healthy", "Leaf miner", "Rust", "Blight"])
            num_per = 48
            d = 1280
            means = rng.normal(0, 1, size=(len(classes), d))
            X = []
            y = []
            for i, cls in enumerate(classes):
                emb = means[i] + 0.4 * rng.normal(0, 1, size=(num_per, d))
                X.append(emb)
                y.extend([cls] * num_per)
            X = np.vstack(X)
            y = np.array(y)
            clf = LogisticRegression(max_iter=300)
            clf.fit(X, y)
            return clf

        @st.cache_resource(show_spinner=False)
        def load_onnx_model(model_path: str):
            if ort is None:
                return None
            if not os.path.exists(model_path):
                return None
            providers = ["CPUExecutionProvider"]
            return ort.InferenceSession(model_path, providers=providers)

        @st.cache_data(show_spinner=False)
        def load_labels(label_path: str):
            if not os.path.exists(label_path):
                return None
            with open(label_path, "r", encoding="utf-8") as f:
                raw = [line.strip() for line in f if line.strip()]
            labels = []
            for s in raw:
                if ":" in s:
                    s = s.split(":", 1)[1]
                s = s.replace("_", " ").strip()
                labels.append(s)
            return labels

        backbone, preprocess, device = load_backbone()
        demo_clf = train_demo_classifier()
        onnx_model = load_onnx_model("/workspace/models/plant_disease.onnx")
        onnx_labels = load_labels("/workspace/models/labels.txt")

        # Optional: allow uploading labels.txt to map class indices to names
        label_file = st.file_uploader("Optional: labels.txt", type=["txt"], accept_multiple_files=False, key="labels_uploader")
        if label_file is not None:
            txt = label_file.read().decode("utf-8", errors="ignore")
            raw = [line.strip() for line in txt.splitlines() if line.strip()]
            labels = []
            for s in raw:
                if ":" in s:
                    s = s.split(":", 1)[1]
                s = s.replace("_", " ").strip()
                labels.append(s)
            st.session_state["onnx_labels_override"] = labels
        if "onnx_labels_override" in st.session_state:
            onnx_labels = st.session_state["onnx_labels_override"]

        # Status caption about model/labels presence
        if onnx_model is not None:
            if onnx_labels is None or len(onnx_labels) == 0:
                st.warning("ONNX model loaded but no labels found. Upload labels.txt to see disease names.")
                st.caption("Model: PlantVillage ONNX — Labels: none loaded")
            else:
                st.caption(f"Model: PlantVillage ONNX — Labels: {len(onnx_labels)} loaded")

        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
            with st.spinner("Analyzing image..."):
                if onnx_model is not None and onnx_labels is not None:
                    tensor = preprocess(img).unsqueeze(0).cpu().numpy().astype(np.float32)
                    input_name = onnx_model.get_inputs()[0].name
                    outputs = onnx_model.run(None, {input_name: tensor})
                    logits = outputs[0][0]
                    exps = np.exp(logits - np.max(logits))
                    probs = exps / exps.sum()
                    top_idx = int(np.argmax(probs))
                    pred_class = onnx_labels[top_idx] if top_idx < len(onnx_labels) else f"Class {top_idx}"
                    pred_conf = float(probs[top_idx])
                    method_note = "PlantVillage ONNX"
                else:
                    tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = backbone(tensor).cpu().numpy()
                    probs_arr = demo_clf.predict_proba(emb)[0]
                    classes = list(demo_clf.classes_)
                    top_idx = int(np.argmax(probs_arr))
                    pred_class = classes[top_idx]
                    pred_conf = float(probs_arr[top_idx])
                    method_note = "MobileNetV2 + Demo classifier (synthetic)"
            st.markdown(
                f"""
                <div class=\"result-box\">
                    <div class=\"metric-title\">Detected pest/disease</div>
                    <div class=\"metric-value\">{pred_class}</div>
                    <div class=\"subtle\">Confidence: {pred_conf*100:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload a plant/leaf photo to classify.")

    with tabs[2]:
        render_supply_chain_tab()

    with st.expander("Preview training data"):
        st.dataframe(data.head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

