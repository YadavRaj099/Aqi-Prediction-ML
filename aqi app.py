import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="AQI Predictor",
    layout="wide"
)

# =====================================================
# MODERN UI STYLES
# =====================================================

st.markdown("""
<style>

body {
    background: linear-gradient(180deg, #020617, #020617);
}

.block-container {
    padding: 2rem 2rem;
}

/* Card UI */
.card {
    background: linear-gradient(180deg,#0f172a,#020617);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #1e293b;
    margin-bottom: 20px;
}

/* Titles */
h1, h2, h3 {
    color: white;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

/* AQI Badge */
.badge {
    padding: 10px 16px;
    border-radius: 10px;
    font-weight: bold;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================

st.markdown("""
<h1>🌍 AI Air Quality Predictor</h1>
<p class="subtitle">
Predict Air Quality Index using machine learning and pollutant levels.
</p>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df = df.dropna()
    return df

data = load_data()

# =====================================================
# MODEL (CACHED)
# =====================================================

@st.cache_resource
def train_model(data):

    features = [
        "PM2.5","PM10","NO","NO2","NOx",
        "NH3","CO","SO2","O3","Benzene",
        "Toluene","Xylene"
    ]

    X = data[features]
    y = data["AQI"]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

model = train_model(data)

# =====================================================
# SESSION STATE
# =====================================================

if "step" not in st.session_state:
    st.session_state.step = 1

# =====================================================
# STEP UI HEADER
# =====================================================

st.progress(st.session_state.step / 3)

# =====================================================
# STEP 1
# =====================================================

if st.session_state.step == 1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📍 Select Location")

    city = st.selectbox(
        "City",
        sorted(data["City"].unique())
    )

    st.session_state.city = city

    if st.button("Next →", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# STEP 2
# =====================================================

elif st.session_state.step == 2:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🧪 Enter Pollution Levels")

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5", 0.0, 1000.0, 50.0)
        pm10 = st.number_input("PM10", 0.0, 1000.0, 80.0)
        no2 = st.number_input("NO2", 0.0, 500.0, 20.0)

    with col2:
        so2 = st.number_input("SO2", 0.0, 500.0, 10.0)
        co = st.number_input("CO", 0.0, 10.0, 0.8)
        o3 = st.number_input("O3", 0.0, 500.0, 30.0)

    st.session_state.inputs = [pm25, pm10, no2, so2, co, o3]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("🚀 Predict AQI", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# STEP 3
# =====================================================

elif st.session_state.step == 3:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📊 AQI Prediction")

    pm25, pm10, no2, so2, co, o3 = st.session_state.inputs

    input_data = np.array([[
        pm25, pm10, 10, no2, 20, 5,
        co, so2, o3, 1, 2, 1
    ]])

    prediction = round(model.predict(input_data)[0], 2)

    # AQI Category
    if prediction <= 50:
        category = "Good"
        color = "#00e400"
    elif prediction <= 100:
        category = "Satisfactory"
        color = "#ffff00"
    elif prediction <= 200:
        category = "Moderate"
        color = "#ff7e00"
    elif prediction <= 300:
        category = "Poor"
        color = "#ff0000"
    elif prediction <= 400:
        category = "Very Poor"
        color = "#8f3f97"
    else:
        category = "Severe"
        color = "#7e0023"

    col1, col2 = st.columns(2)

    col1.metric("AQI", prediction)
    col2.markdown(
        f"<div class='badge' style='background:{color};color:black'>{category}</div>",
        unsafe_allow_html=True
    )

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        gauge={'axis': {'range': [0, 500]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Advisory
    st.subheader("Health Advisory")

    if prediction < 100:
        st.success("Air quality is acceptable.")
    elif prediction < 200:
        st.warning("Sensitive individuals should limit outdoor activity.")
    else:
        st.error("Avoid prolonged exposure.")

    if st.button("Restart", use_container_width=True):
        st.session_state.step = 1
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("Built with Machine Learning • AQI Prediction System")