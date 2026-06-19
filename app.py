# =============================================================================
# CARGO BREAKDOWN TIME PREDICTION SYSTEM
# Jeddah Airport — Aviation Logistics Intelligence Platform
# =============================================================================
# Model  : AutoGluon TabularPredictor (WeightedEnsemble_L3)
# Target : Breakdown_Time_Minutes
# Selector: Flight_Number from test_dataset.csv — fields auto-populate
#
# Drop before predict: Unnamed: 0, Flight_Number, Arrival_Date,
#                      Breakdown_Time_Minutes
# AutoGluon handles ALL preprocessing — no encoding / scaling needed.
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cargo Breakdown Time Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)
        
# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AutoGluon model…")
def load_model():
    import os
    import zipfile
    from autogluon.tabular import TabularPredictor
    MODEL_DIR = "AutogluonModels/ag-20260618_062203"
    ZIP_FILE = "ag-20260618_062203.zip"
    if not os.path.exists(MODEL_DIR):
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall("AutoGluonModels")
    predictor = TabularPredictor.load(MODEL_DIR)
    return predictor

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    dataset   = pd.read_csv("dataset/cargo_breakdown_dataset_30000.csv")
    test_data = pd.read_csv("dataset/test_dataset.csv")
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in test_data.columns:
        test_data = test_data.drop(columns=["Unnamed: 0"])
    return dataset, test_data

model_ok = False
data_ok  = False
predictor = None
dataset   = None
test_data = None

try:
    predictor = load_model()
    model_ok  = True
except Exception as _me:
    pass  # handled gracefully in prediction section

try:
    dataset, test_data = load_data()
    data_ok = True
except Exception as _de:
    st.error(f"Dataset not found: {_de}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — matches the reference screenshot exactly
# White background, clean light theme, blue active tab
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Global layout ── */
.block-container {
    padding-top    : 1rem;
    padding-bottom : 0rem;
}
.main { background-color: #ffffff; }
body, .stApp { background-color: #ffffff; }

/* ── Typography ── */
h1 { font-size: 2rem !important; font-weight: 800 !important;
     color: #111827 !important; margin-bottom: 2px !important; }
p  { color: #6B7280; font-size: 15px; margin-bottom: 0; }

/* ── Section headers (bold, underlined in blue) ── */
.section-header {
    font-size      : 15px;
    font-weight    : 700;
    color          : #111827;
    border-bottom  : 2px solid #1565ff;
    padding-bottom : 6px;
    margin-bottom  : 14px;
    margin-top     : 10px;
}

/* ── Sub-section bold label ── */
.sub-label {
    font-weight  : 700;
    font-size    : 14px;
    color        : #111827;
    margin-bottom: 10px;
}

/* ── Tabs — match screenshot exactly ── */
.stTabs [data-baseweb="tab-list"] {
    gap              : 0px;
    background-color : transparent;
    border-bottom    : 1px solid #e5e7eb;
    padding-bottom   : 0;
    margin-bottom    : 0;
}
.stTabs [data-baseweb="tab"] {
    height        : 40px;
    padding       : 0 20px;
    border-radius : 0;
    font-size     : 14px;
    font-weight   : 500;
    color         : #374151;
    background    : transparent;
    border        : none;
    border-bottom : 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background    : #1565ff !important;
    color         : white   !important;
    border-radius : 4px 4px 0 0 !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {
    color: white !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color      : #111827;
    background : #f3f4f6;
    border-radius: 4px 4px 0 0;
}
/* Remove tab panel top border */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}

/* ── Disabled / read-only text inputs (grey filled boxes) ── */
.stTextInput input:disabled,
.stTextInput input[disabled] {
    background-color : #f3f4f6 !important;
    color            : #374151 !important;
    border           : 1px solid #e5e7eb !important;
    border-radius    : 6px !important;
    font-size        : 14px !important;
    height           : 40px;
    cursor           : default;
    -webkit-text-fill-color: #374151 !important;
    opacity: 1 !important;
}
.stTextInput label {
    font-size   : 13px !important;
    color       : #6B7280 !important;
    font-weight : 400 !important;
    margin-bottom: 3px;
}

/* ── Selectbox (Flight Number) ── */
.stSelectbox label {
    font-size   : 13px !important;
    color       : #6B7280 !important;
    font-weight : 400 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color : #f3f4f6 !important;
    border           : 1px solid #e5e7eb !important;
    border-radius    : 6px !important;
    min-height       : 40px !important;
    font-size        : 14px !important;
}

/* ── Predict button — small, not full-width ── */
.predict-btn .stButton > button {
    background-color : #ffffff;
    color            : #111827;
    border           : 1px solid #d1d5db;
    border-radius    : 6px;
    height           : 36px;
    width            : auto !important;
    min-width        : 160px;
    font-size        : 13px;
    font-weight      : 500;
    padding          : 0 16px;
    transition       : all 0.15s;
    margin-top       : 0;
}
.predict-btn .stButton > button:hover {
    background-color: #f3f4f6;
    border-color    : #9ca3af;
}

/* ── Download / action buttons ── */
.stDownloadButton > button {
    background-color : #1565ff;
    color            : white;
    border-radius    : 6px;
    font-size        : 13px;
    font-weight      : 600;
    border           : none;
    padding          : 0 20px;
    height           : 38px;
    transition       : background 0.15s;
}

.stDownloadButton > button * {
    color : #ffffff !important;
    fill  : #ffffff !important;
}
.stDownloadButton > button:hover { background-color: #1251d3; }

/* ── Run bulk prediction button ── */
.bulk-btn .stButton > button {
    background-color : #1565ff;
    color            : white;
    border-radius    : 6px;
    height           : 40px;
    font-size        : 14px;
    font-weight      : 600;
    border           : none;
    padding          : 0 24px;
    width            : auto !important;
    min-width        : 200px;
    box-shadow       : 0 1px 4px rgba(21,101,255,0.3);
}
.bulk-btn .stButton > button:hover { background-color: #1251d3; }

/* ── KPI Metric Cards ── */
[data-testid="stMetric"] {
    background    : #ffffff;
    padding       : 16px 18px;
    border-radius : 10px;
    border        : 1px solid #e5e7eb;
    border-top    : 4px solid #1565ff;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] {
    font-size   : 26px;
    font-weight : 700;
    color       : #111827;
}
[data-testid="stMetricLabel"] {
    font-size  : 12px;
    color      : #6B7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Prediction result cards ── */
.pred-card-green {
    background    : linear-gradient(135deg, #f0fdf4, #dcfce7);
    padding       : 24px 28px;
    border-radius : 12px;
    border-left   : 5px solid #16a34a;
    border        : 1px solid #bbf7d0;
    border-left   : 5px solid #16a34a;
    margin-bottom : 16px;
}
.pred-card-yellow {
    background    : linear-gradient(135deg, #fefce8, #fef9c3);
    padding       : 24px 28px;
    border-radius : 12px;
    border        : 1px solid #fde68a;
    border-left   : 5px solid #ca8a04;
    margin-bottom : 16px;
}
.pred-card-red {
    background    : linear-gradient(135deg, #fff1f2, #fee2e2);
    padding       : 24px 28px;
    border-radius : 12px;
    border        : 1px solid #fecaca;
    border-left   : 5px solid #dc2626;
    margin-bottom : 16px;
}
.pred-title {
    font-size      : 11px;
    font-weight    : 700;
    color          : #6B7280;
    text-transform : uppercase;
    letter-spacing : 0.1em;
    margin-bottom  : 8px;
}
.pred-value {
    font-size   : 48px;
    font-weight : 800;
    color       : #111827;
    line-height : 1;
}
.pred-unit  { font-size: 18px; font-weight: 400; color: #6B7280; margin-left: 4px; }
.pred-sub   { font-size: 17px; color: #374151; margin-top: 10px; }
.pred-risk  { font-size: 17px; font-weight: 700; margin-top: 12px; }
.risk-green  { color: #16a34a; }
.risk-yellow { color: #ca8a04; }
.risk-red    { color: #dc2626; }

/* ── Divider ── */
hr { border-color: #e5e7eb; margin: 10px 0 18px 0; }

/* ── Table ── */
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #e5e7eb; }

/* ── Remove default top padding on inputs in columns ── */
.stColumn .stTextInput { margin-bottom: 4px; }

/* ── Model insight card ── */
.model-card {
    background    : #f9fafb;
    border        : 1px solid #e5e7eb;
    border-radius : 10px;
    padding       : 16px 18px;
    text-align    : center;
    margin-bottom : 8px;
}
.model-card-label {
    font-size     : 11px;
    font-weight   : 700;
    color         : #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom : 6px;
}
.model-card-value {
    font-size  : 22px;
    font-weight: 800;
    color      : #1565ff;
}
.model-card-unit {
    font-size : 12px;
    color     : #6B7280;
    margin-top: 2px;
}

/* ── Plotly chart white background ── */
.js-plotly-plot { border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# APP HEADER  — plain title + subtitle, like reference
# ─────────────────────────────────────────────────────────────
st.markdown("""
<h1>Cargo Breakdown Time Prediction System</h1>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
TARGET     = "Breakdown_Time_Minutes"
DROP_COLS  = ["Unnamed: 0", "Flight_Number", "Arrival_Date", TARGET]

FEATURE_COLS = [
    "Year", "Month", "Season", "Origin", "Destination", "Flight_Type",
    "Aircraft_Type", "Shipment_Count", "Cargo_Weight_KG", "Cargo_Volume_CBM",
    "ULD_Count", "ULD_Type", "Pallet_Count", "Nature_of_Goods",
    "Cargo_Mix_Complexity", "Manpower_Assigned", "Forklift_Count",
    "Equipment_Availability", "Shift", "Weather_Condition",
    "Customs_Hold_Units", "Storage_Distance_Meters", "Breakdown_Priority"
]

# Shared Plotly layout — white theme matching the reference app
CHART_LAYOUT = dict(
    template      = "plotly_white",
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    height        = 420,
    margin        = dict(l=20, r=20, t=55, b=20),
    title_font    = dict(size=15, color="#111827"),
    font          = dict(color="#374151", size=12),
)
PRIMARY = "#1565ff"
BLUES   = ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd","#bfdbfe","#dbeafe"]


def classify_risk(minutes: float):
    """Return (label, card_class, risk_class)."""
    if minutes <= 60:
        return "Low Processing Time",    "pred-card-green",  "risk-green", 
    elif minutes <= 120:
        return "Medium Processing Time", "pred-card-yellow", "risk-yellow",
    else:
        return "High Processing Time",   "pred-card-red",    "risk-red",


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Breakdown Time Prediction",
    "Analytics Dashboard",
    "Bulk Breakdown Time Analysis",
])

# =============================================================================
# TAB 1 — BREAKDOWN TIME PREDICTION
# =============================================================================
with tab1:

    # ── SELECT FLIGHT ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Select Flight</div>",
                unsafe_allow_html=True)

    flight_list = test_data["Flight_Number"].unique().tolist()

    selected_flight = st.selectbox(
        "Flight Number",
        flight_list,
        help="Select a Flight Number — all fields auto-populate below"
    )

    # Fetch the row for this flight
    row = test_data[test_data["Flight_Number"] == selected_flight].iloc[0]

    # ── SHIPMENT DETAILS ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Shipment Details</div>",
                unsafe_allow_html=True)

    left_col, right_col = st.columns(2)

    # ── LEFT : Shipment Information ───────────────────────────────────────────
    with left_col:
        st.markdown("<div class='sub-label'>Shipment Information</div>",
                    unsafe_allow_html=True)

        st.text_input("Arrival Date",      value=str(row["Arrival_Date"]),  disabled=True)
        st.text_input("Origin",            value=str(row["Origin"]),         disabled=True)
        st.text_input("Destination",       value=str(row["Destination"]),    disabled=True)
        st.text_input("Flight Type",       value=str(row["Flight_Type"]),    disabled=True)
        st.text_input("Aircraft Type",     value=str(row["Aircraft_Type"]),  disabled=True)
        st.text_input("Shipment Count",    value=str(row["Shipment_Count"]), disabled=True)
        st.text_input("Cargo Weight (KG)", value=str(row["Cargo_Weight_KG"]),disabled=True)
        st.text_input("Cargo Volume (CBM)",value=str(row["Cargo_Volume_CBM"]),disabled=True)
        st.text_input("Nature of Goods",   value=str(row["Nature_of_Goods"]),disabled=True)
        st.text_input("Cargo Mix Complexity", value=str(row["Cargo_Mix_Complexity"]), disabled=True)
        st.text_input("Season",            value=str(row["Season"]),         disabled=True)

    # ── RIGHT : Operations Information ───────────────────────────────────────
    with right_col:
        st.markdown("<div class='sub-label'>Operations Information</div>",
                    unsafe_allow_html=True)

        st.text_input("ULD Count",               value=str(row["ULD_Count"]),              disabled=True)
        st.text_input("ULD Type",                value=str(row["ULD_Type"]),               disabled=True)
        st.text_input("Pallet Count",            value=str(row["Pallet_Count"]),           disabled=True)
        st.text_input("Manpower Assigned",       value=str(row["Manpower_Assigned"]),      disabled=True)
        st.text_input("Forklift Count",          value=str(row["Forklift_Count"]),         disabled=True)
        st.text_input("Equipment Availability",  value=str(row["Equipment_Availability"]), disabled=True)
        st.text_input("Shift",                   value=str(row["Shift"]),                  disabled=True)
        st.text_input("Weather Condition",       value=str(row["Weather_Condition"]),      disabled=True)
        st.text_input("Customs Hold Units",      value=str(row["Customs_Hold_Units"]),     disabled=True)
        st.text_input("Storage Distance (m)",    value=str(row["Storage_Distance_Meters"]),disabled=True)
        st.text_input("Breakdown Priority",      value=str(row["Breakdown_Priority"]),     disabled=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PREDICT BUTTON — small, left-aligned ─────────────────────────────────
    st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Breakdown Time", key="predict_single")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── PREDICTION RESULT ─────────────────────────────────────────────────────
    if predict_clicked:
        if not model_ok:
            st.error(
                f"AutoGluon model not found.\n"
                f"Expected path: {MODEL_DIR}"
            )
        else:
            try:
                # Build input — keep only feature columns, AutoGluon handles the rest
                input_df = test_data[test_data["Flight_Number"] == selected_flight].copy()
                input_df = input_df.drop(
                    columns=[c for c in DROP_COLS if c in input_df.columns],
                    errors="ignore"
                )
                input_df = input_df[FEATURE_COLS]

                # Predict
                prediction = float(predictor.predict(input_df).iloc[0])
                hrs  = int(prediction // 60)
                mins = int(prediction % 60)

                risk_label, card_class, risk_class = classify_risk(prediction)

                # ── Result layout ────────────────────────────────────────────
                res_col, factor_col = st.columns([1, 1])

                with res_col:
                    st.markdown(
                        "<div class='section-header'>Prediction Result</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"""
                    <div class='{card_class}'>
                        <div class='pred-title'>Predicted Breakdown Time</div>
                        <div class='pred-value'>
                            {prediction:.0f}
                            <span class='pred-unit'>Minutes</span>
                        </div>
                        <div class='pred-sub'>
                            {hrs} Hour{'s' if hrs != 1 else ''} {mins:02d} Minutes
                        </div>
                        <div class='pred-risk {risk_class}'> {risk_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with factor_col:
                    st.markdown(
                        "<div class='section-header'>Key Factors</div>",
                        unsafe_allow_html=True
                    )
                    kf = pd.DataFrame({
                        "Factor": [
                            "Cargo Weight (KG)", "Cargo Volume (CBM)", "ULD Type",
                            "ULD Count", "Shipment Count", "Manpower Assigned",
                            "Forklift Count", "Customs Hold Units",
                            "Equipment Availability", "Weather", "Shift", "Priority"
                        ],
                        "Value": [
                            str(row["Cargo_Weight_KG"]),
                            str(row["Cargo_Volume_CBM"]),
                            str(row["ULD_Type"]),
                            str(row["ULD_Count"]),
                            str(row["Shipment_Count"]),
                            str(row["Manpower_Assigned"]),
                            str(row["Forklift_Count"]),
                            str(row["Customs_Hold_Units"]),
                            str(row["Equipment_Availability"]),
                            str(row["Weather_Condition"]),
                            str(row["Shift"]),
                            str(row["Breakdown_Priority"])
                        ]
                    })
                    st.dataframe(kf, use_container_width=True,
                                 hide_index=True, height=380)

                st.session_state["last_prediction"] = prediction

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    # ── MODEL INSIGHTS + FEATURE IMPORTANCE ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model Insights</div>",
                unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    insight_cards = [
        ("Best Model",  "WeightedEnsemble L3",  "Stacked Ensemble"),
        ("R² Score",    "0.9835",                "98.3% Variance"),
        ("MAE",         "6.33 min",             "Mean Absolute Error")
    ]
    for col, (label, val, unit) in zip([mc1, mc2, mc3], insight_cards):
        with col:
            st.markdown(f"""
            <div class='model-card'>
                <div class='model-card-label'>{label}</div>
                <div class='model-card-value'>{val}</div>
                <div class='model-card-unit'>{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TAB 2 — ANALYTICS DASHBOARD
# =============================================================================
with tab2:

    st.markdown("<div class='section-header'>Executive Analytics Dashboard</div>",
                unsafe_allow_html=True)

    dash = dataset.copy()
    if "Arrival_Date" in dash.columns:
        dash["Arrival_Date"] = pd.to_datetime(dash["Arrival_Date"], errors="coerce")
        ds_min = dash["Arrival_Date"].min().date()
        ds_max = dash["Arrival_Date"].max().date()
    else:
        ds_min = ds_max = None

    # ── Date filter row ───────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        from_date = st.date_input("From Date", value=ds_min,
                                  min_value=ds_min, max_value=ds_max)
    with fc2:
        to_date = st.date_input("To Date", value=ds_max,
                                min_value=ds_min, max_value=ds_max)
    with fc3:
        quick = st.selectbox("Quick Filter",
                             ["All Time","Last 30 Days","Last 90 Days",
                              "Last 6 Months","Last 1 Year"])

    if "Arrival_Date" in dash.columns:
        max_ts = dash["Arrival_Date"].max()
        if quick == "Last 30 Days":
            from_date = (max_ts - pd.Timedelta(days=30)).date()
        elif quick == "Last 90 Days":
            from_date = (max_ts - pd.Timedelta(days=90)).date()
        elif quick == "Last 6 Months":
            from_date = (max_ts - pd.DateOffset(months=6)).date()
        elif quick == "Last 1 Year":
            from_date = (max_ts - pd.DateOffset(years=1)).date()

        filtered = dash[
            (dash["Arrival_Date"] >= pd.Timestamp(from_date)) &
            (dash["Arrival_Date"] <= pd.Timestamp(to_date))
        ].copy()
    else:
        filtered = dash.copy()

    if filtered.empty:
        st.warning("No data in the selected date range.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        ("Total Shipments",    f"{len(filtered):,}"),
        ("Avg Breakdown Time", f"{filtered[TARGET].mean():.1f} min"),
        ("Max Breakdown Time", f"{filtered[TARGET].max():.0f} min"),
        ("Min Breakdown Time", f"{filtered[TARGET].min():.0f} min"),
        ("Avg Cargo Weight",   f"{filtered['Cargo_Weight_KG'].mean():,.0f} KG"),
        ("Avg ULD Count",      f"{filtered['ULD_Count'].mean():.1f}"),
    ]
    for col, (lbl, val) in zip([k1, k2, k3, k4, k5, k6], kpis):
        with col:
            st.metric(lbl, val)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 1 ─────────────────────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        if "Arrival_Date" in filtered.columns:
            monthly = (
                filtered.groupby(filtered["Arrival_Date"].dt.to_period("M"))[TARGET]
                .mean().reset_index()
            )
            monthly["Arrival_Date"] = monthly["Arrival_Date"].dt.to_timestamp()
            monthly[TARGET] = monthly[TARGET].round(1)
            fig1 = px.line(monthly, x="Arrival_Date", y=TARGET, markers=True,
                           title="1. Monthly Avg Breakdown Time Trend")
            fig1.update_traces(
                mode="lines+markers+text",
                text=monthly[TARGET], textposition="top center",
                line=dict(color=PRIMARY, width=3),
                marker=dict(size=8, color=PRIMARY),
                textfont=dict(size=10)
            )
            fig1.update_layout(
                **CHART_LAYOUT,
                xaxis=dict(title="Month", showgrid=False, tickfont=dict(size=12)),
                yaxis=dict(title="Avg Breakdown Time (min)",
                           gridcolor="#E5E7EB", tickfont=dict(size=12))
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with r1c2:
        goods = (
            filtered.groupby("Nature_of_Goods")[TARGET]
            .mean().reset_index().sort_values(TARGET, ascending=False)
        )
        goods[TARGET] = goods[TARGET].round(1)
        fig2 = px.bar(goods, x="Nature_of_Goods", y=TARGET, text=TARGET,
                      title="2. Cargo Type vs Avg Breakdown Time")
        fig2.update_traces(
            marker_color=BLUES[:len(goods)],
            texttemplate="%{text:.1f}", textposition="outside"
        )
        fig2.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Cargo Type", tickangle=-15),
            yaxis=dict(title="Avg Breakdown Time (min)", gridcolor="#E5E7EB",
                       range=[0, goods[TARGET].max() + 25]),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 2 ─────────────────────────────────────────────────────────────────
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        uld = (
            filtered.groupby("ULD_Type")[TARGET]
            .mean().reset_index().sort_values(TARGET, ascending=True)
        )
        uld[TARGET] = uld[TARGET].round(1)
        fig3 = px.bar(uld, x=TARGET, y="ULD_Type", orientation="h", text=TARGET,
                      color=TARGET, color_continuous_scale="Blues",
                      title="3. ULD Type vs Breakdown Time")
        fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(
            **CHART_LAYOUT, coloraxis_showscale=False,
            xaxis=dict(title="Avg Breakdown Time (min)", gridcolor="#E5E7EB"),
            yaxis=dict(title="ULD Type")
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with r2c2:
        shift = (
            filtered.groupby("Shift")[TARGET].mean().reset_index()
        )
        shift[TARGET] = shift[TARGET].round(1)
        fig4 = px.pie(shift, names="Shift", values=TARGET, hole=0.5,
                      color_discrete_sequence=BLUES,
                      title="4. Shift-wise Breakdown Performance")
        fig4.update_traces(
            textposition="outside", textinfo="percent+label",
            pull=[0.03] + [0] * (len(shift) - 1),
            marker=dict(line=dict(color="white", width=2))
        )
        fig4.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    with r2c3:
        season = (
            filtered.groupby("Season")[TARGET]
            .mean().reset_index().sort_values(TARGET, ascending=False)
        )
        season[TARGET] = season[TARGET].round(1)
        fig5 = px.bar(season, x="Season", y=TARGET, text=TARGET,
                      title="5. Season-wise Breakdown Analysis",
                      color=TARGET, color_continuous_scale="Blues")
        fig5.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig5.update_layout(
            **CHART_LAYOUT, coloraxis_showscale=False,
            xaxis=dict(title="Season"),
            yaxis=dict(title="Avg Breakdown Time (min)", gridcolor="#E5E7EB",
                       range=[0, season[TARGET].max() + 25])
        )
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 3 ─────────────────────────────────────────────────────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        fig6 = px.scatter(
            sample, x="Cargo_Weight_KG", y=TARGET,
            color="Flight_Type", color_discrete_sequence=BLUES,
            title="6. Cargo Weight vs Breakdown Time", opacity=0.55
        )
        fig6.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Cargo Weight (KG)", gridcolor="#E5E7EB"),
            yaxis=dict(title="Breakdown Time (min)", gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

    with r3c2:
        mp_bins = filtered.copy()
        mp_bins["Manpower_Group"] = pd.cut(
            mp_bins["Manpower_Assigned"], bins=[0, 5, 10, 15, 20],
            labels=["2–5", "6–10", "11–15", "16–20"]
        )
        mp = mp_bins.groupby("Manpower_Group", observed=True)[TARGET].mean().reset_index()
        mp[TARGET] = mp[TARGET].round(1)
        fig7 = px.bar(mp, x="Manpower_Group", y=TARGET, text=TARGET,
                      title="7. Manpower Assigned vs Breakdown Time",
                      color=TARGET, color_continuous_scale="Blues")
        fig7.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig7.update_layout(
            **CHART_LAYOUT, coloraxis_showscale=False,
            xaxis=dict(title="Manpower Group"),
            yaxis=dict(title="Avg Breakdown Time (min)", gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})

    # ── ROW 4 ─────────────────────────────────────────────────────────────────
    r4c1, r4c2, r4c3 = st.columns(3)

    with r4c1:
        cb = filtered.copy()
        cb["Customs_Group"] = pd.cut(
            cb["Customs_Hold_Units"], bins=[-1, 3, 7, 11, 15],
            labels=["0–3", "4–7", "8–11", "12–15"]
        )
        cust = cb.groupby("Customs_Group", observed=True)[TARGET].mean().reset_index()
        cust[TARGET] = cust[TARGET].round(1)
        fig8 = px.bar(
            cust.sort_values(TARGET, ascending=False),
            x="Customs_Group", y=TARGET, text=TARGET,
            title="8. Customs Hold Impact",
            color=TARGET, color_continuous_scale="BLUES"
        )
        fig8.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig8.update_layout(
            **CHART_LAYOUT, coloraxis_showscale=False,
            xaxis=dict(title="Customs Hold Units"),
            yaxis=dict(title="Avg Breakdown Time (min)", gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig8, use_container_width=True, config={"displayModeBar": False})

    with r4c2:
        ft = (
            filtered.groupby("Flight_Type")[TARGET]
            .agg(["mean","min","max"]).reset_index()
            .rename(columns={"mean":"Avg","min":"Min","max":"Max"})
        )
        fig9 = go.Figure()
        fig9.add_trace(go.Bar(name="Avg", x=ft["Flight_Type"],
                              y=ft["Avg"].round(1),
                              text=ft["Avg"].round(1), textposition="outside",
                              marker_color=BLUES[2]))
        fig9.add_trace(go.Bar(name="Min", x=ft["Flight_Type"],
                              y=ft["Min"], marker_color=BLUES[4]))
        fig9.add_trace(go.Bar(name="Max", x=ft["Flight_Type"],
                              y=ft["Max"], marker_color=BLUES[0]))
        fig9.update_layout(
            **CHART_LAYOUT, barmode="group",
            title="9. Flight Type Performance",
            xaxis=dict(title="Flight Type"),
            yaxis=dict(title="Breakdown Time (min)", gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig9, use_container_width=True, config={"displayModeBar": False})

    with r4c3:
        eq = (
            filtered.groupby("Equipment_Availability")[TARGET]
            .mean().reset_index().sort_values(TARGET, ascending=False)
        )
        eq[TARGET] = eq[TARGET].round(1)
        fig10 = px.bar(
            eq, x="Equipment_Availability", y=TARGET, text=TARGET,
            title="10. Equipment Availability Impact",
            color="Equipment_Availability",
            color_discrete_map={"Low":"#0d47a1","Medium":"#1976d2","High":"#64b5f6"}
        )
        fig10.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig10.update_layout(
            **CHART_LAYOUT, showlegend=False,
            xaxis=dict(title="Equipment Availability"),
            yaxis=dict(title="Avg Breakdown Time (min)", gridcolor="#E5E7EB",
                       range=[0, eq[TARGET].max() + 25])
        )
        st.plotly_chart(fig10, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# TAB 3 — BULK BREAKDOWN TIME ANALYSIS
# =============================================================================
with tab3:

    st.markdown("<div class='section-header'>Bulk Breakdown Time Analysis</div>",
                unsafe_allow_html=True)

    # ── Info + template download ──────────────────────────────────────────────
    st.info(
        "Upload a CSV file containing the same 23 feature columns used during training. "
        "Extra columns (Flight_Number, Arrival_Date) are automatically ignored."
    )

    TEMPLATE_COLS = FEATURE_COLS
    sample_rows = [
        [2024,7,"Hajj","Dubai","Jeddah","International","B747F",
         600,28000,350,12,"AKE",15,"General Cargo","Medium",10,4,"Medium","Morning","Clear",8,200,"Urgent"],
        [2024,3,"Ramadan","Doha","Jeddah","International","B777F",
         900,42000,450,20,"PAG",8,"Perishable","High",14,6,"High","Evening","Foggy",5,120,"Critical"],
        [2023,8,"Normal","Medina","Jeddah","Domestic","A320",
         120,4500,30,3,"PLA",6,"Pharma","Low",18,2,"Low","Night","Clear",12,40,"Normal"],
    ]
    tpl_df  = pd.DataFrame(sample_rows, columns=TEMPLATE_COLS)
    tpl_csv = tpl_df.to_csv(index=False)

    dl_col, _ = st.columns([2, 5])
    with dl_col:
        st.download_button(
            "Download CSV Template", data=tpl_csv,
            file_name="breakdown_template.csv", mime="text/csv"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload Shipment CSV", type=["csv"])

    bulk_raw = None
    if uploaded is not None:
        try:
            bulk_raw = pd.read_csv(uploaded)
            st.success(f"File loaded: **{len(bulk_raw):,} rows** · **{len(bulk_raw.columns)} columns**")
            st.dataframe(bulk_raw.head(5), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            bulk_raw = None

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run bulk prediction ───────────────────────────────────────────────────
    st.markdown("<div class='bulk-btn'>", unsafe_allow_html=True)
    run_bulk = st.button("Run Bulk Prediction", key="bulk_predict",
                         disabled=(not model_ok))
    st.markdown("</div>", unsafe_allow_html=True)

    if run_bulk:
        if bulk_raw is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                drop = ["Unnamed: 0", "Flight_Number", "Arrival_Date", TARGET]
                bulk_input = bulk_raw.drop(
                    columns=[c for c in drop if c in bulk_raw.columns],
                    errors="ignore"
                )
                missing = [c for c in FEATURE_COLS if c not in bulk_input.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    with st.spinner("Running AutoGluon predictions…"):
                        preds = predictor.predict(bulk_input[FEATURE_COLS])

                    results = bulk_raw.copy()
                    results["Predicted_Breakdown_Time_Min"] = np.round(preds.values, 1)
                    results["Risk_Category"] = results["Predicted_Breakdown_Time_Min"].apply(
                        lambda v: "Low" if v <= 60 else ("Medium" if v <= 120 else "High")
                    )
                    st.session_state["bulk_results"] = results
                    st.success(f"Prediction complete for {len(results):,} records.")

            except Exception as e:
                st.error(f"Bulk prediction failed: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if "bulk_results" in st.session_state:
        results = st.session_state["bulk_results"]

        bk1, bk2, bk3, bk4, bk5 = st.columns(5)
        bk_kpis = [
            ("Total Records",     f"{len(results):,}"),
            ("Avg Predicted",     f"{results['Predicted_Breakdown_Time_Min'].mean():.1f} min"),
            ("Max Predicted",     f"{results['Predicted_Breakdown_Time_Min'].max():.0f} min"),
            ("Min Predicted",     f"{results['Predicted_Breakdown_Time_Min'].min():.0f} min"),
            ("High Risk Count",   f"{(results['Risk_Category']=='High').sum():,}"),
        ]
        for col, (lbl, val) in zip([bk1, bk2, bk3, bk4, bk5], bk_kpis):
            with col:
                st.metric(lbl, val)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk filter + table
        rf_col, _ = st.columns([2, 5])
        with rf_col:
            risk_filter = st.selectbox(
                "Filter by Risk Category",
                ["All", "Low", "Medium", "High"],
                key="risk_filter"
            )

        display = (results if risk_filter == "All"
                   else results[results["Risk_Category"] == risk_filter])

        st.info(f"Showing **{len(display):,}** records — Risk: **{risk_filter}**")

        id_cols   = [c for c in ["Flight_Number", "Arrival_Date"] if c in display.columns]
        show_cols = id_cols + ["Predicted_Breakdown_Time_Min", "Risk_Category"]
        show_cols = [c for c in show_cols if c in display.columns]

        st.dataframe(
            display[show_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        # Download
        dl2_col, _ = st.columns([2, 5])
        with dl2_col:
            st.download_button(
                "Download Predictions CSV",
                data=display.to_csv(index=False),
                file_name="cargo_breakdown_predictions.csv",
                mime="text/csv"
            )

        # Distribution chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig_h = px.histogram(
            results, x="Predicted_Breakdown_Time_Min", nbins=40,
            color="Risk_Category",
            color_discrete_map={"Low":"#16a34a","Medium":"#ca8a04","High":"#dc2626"},
            title="Breakdown Time Distribution — All Predictions",
            labels={"Predicted_Breakdown_Time_Min":"Predicted Breakdown Time (min)"}
        )
        fig_h.add_vline(x=60,  line_dash="dash", line_color="#16a34a",
                        annotation_text="60 min threshold",
                        annotation_position="top right")
        fig_h.add_vline(x=120, line_dash="dash", line_color="#dc2626",
                        annotation_text="120 min threshold",
                        annotation_position="top right")
        fig_h.update_layout(
            **{**CHART_LAYOUT, "height":380},
            xaxis=dict(title="Predicted Breakdown Time (min)", gridcolor="#E5E7EB"),
            yaxis=dict(title="Number of Records", gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})