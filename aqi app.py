import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="AQI Predictor", layout="wide")

# =====================================================
# STYLES (FIXED SIDEBAR + UI)
# =====================================================

st.markdown("""
<style>

/* Main background */
body { background: #020617; }

/* Sidebar full styling */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
    padding-top: 20px;
}

/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}

/* Sidebar buttons */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg,#0f172a,#020617);
    border: 1px solid #334155;
    border-radius: 12px;
    color: white;
    padding: 10px;
    margin-bottom: 10px;
    font-weight: 600;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: #0ea5e9;
    color: black;
}

/* Active button */
.active-btn {
    background: #0ea5e9 !important;
    color: black !important;
}

/* Metric box */
.info-box {
    background: linear-gradient(135deg,#1e293b,#020617);
    padding:20px;
    border-radius:12px;
    border:1px solid #334155;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HUMAN STATE FUNCTION
# =====================================================

def get_human_state(aqi):
    if aqi <= 50:
        return "😊", "Healthy", "#22c55e"
    elif aqi <= 100:
        return "🙂", "Slightly Affected", "#eab308"
    elif aqi <= 200:
        return "😷", "Mask Recommended", "#f97316"
    elif aqi <= 300:
        return "🤧", "Breathing Issues", "#ef4444"
    elif aqi <= 400:
        return "🫁", "Serious Condition", "#a855f7"
    else:
        return "🚑", "Emergency / ICU Risk", "#7f1d1d"

# =====================================================
# SIDEBAR NAVIGATION (FIXED)
# =====================================================

st.sidebar.markdown('<div class="sidebar-title">🌍 AI AQI App</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

if "page" not in st.session_state:
    st.session_state.page = "Predictor"

# Buttons
predictor_btn = st.sidebar.button("🔮 Predictor")
analytics_btn = st.sidebar.button("📊 Analytics")

if predictor_btn:
    st.session_state.page = "Predictor"

if analytics_btn:
    st.session_state.page = "Analytics"

# Current page indicator
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 👉 Current: {st.session_state.page}")
st.sidebar.markdown("---")
st.sidebar.success("✨ AI Powered System")

page = st.session_state.page

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

data = load_data()

# =====================================================
# MODEL
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    return model, score

model, score = train_model(data)

# =====================================================
# HEADER
# =====================================================

st.title("🌍 AI Air Quality Predictor")

st.markdown(f"""
<div class="info-box">
<b>Model:</b> Random Forest Regressor<br>
<b>Accuracy (R²):</b> {round(score,2)}<br>
<b>Cities:</b> {data['City'].nunique()}
</div>
""", unsafe_allow_html=True)

# =====================================================
# PREDICTOR PAGE
# =====================================================

if page == "Predictor":

    st.subheader("📍 Select City")

    city = st.selectbox("City", sorted(data["City"].unique()))

    st.subheader("🧪 Pollution Levels")

    pm25 = st.number_input("PM2.5", 0.0, 1000.0, 50.0)
    pm10 = st.number_input("PM10", 0.0, 1000.0, 80.0)
    no2 = st.number_input("NO2", 0.0, 500.0, 20.0)
    so2 = st.number_input("SO2", 0.0, 500.0, 10.0)
    co = st.number_input("CO", 0.0, 10.0, 0.8)
    o3 = st.number_input("O3", 0.0, 500.0, 30.0)

    if st.button("🚀 Predict AQI"):

        input_data = np.array([[pm25, pm10, 10, no2, 20, 5, co, so2, o3, 1, 2, 1]])
        prediction = round(model.predict(input_data)[0], 2)

        emoji, state, color = get_human_state(prediction)

        col1, col2 = st.columns(2)

        col1.metric("AQI", prediction)

        col2.markdown(f"""
        <div style="text-align:center; padding:15px; background:{color}; border-radius:10px;">
        <h1>{emoji}</h1>
        <b>{state}</b>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            gauge={'axis': {'range': [0, 500]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    st.subheader("👤 AQI Impact Simulator")

    aqi_input = st.slider("AQI Level", 0, 500, 100)

    emoji, state, color = get_human_state(aqi_input)

    st.markdown(f"""
    <div style="text-align:center; padding:20px; background:{color}; border-radius:12px;">
    <h1>{emoji}</h1>
    <b>{state}</b>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Trend")

    pollutant = st.selectbox("Pollutant",
        ["PM2.5","PM10","NO2","SO2","CO","O3"])

    trend = data.groupby("Date")[pollutant].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("Built with Machine Learning • AQI Prediction System")