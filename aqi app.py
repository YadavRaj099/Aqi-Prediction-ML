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
# STYLES
# =====================================================

st.markdown("""
<style>
body { background: #020617; }

.block-container { padding: 2rem; }

.badge {
    padding: 12px 18px;
    border-radius: 12px;
    font-weight: bold;
    display: inline-block;
}

div.stButton > button {
    background: linear-gradient(135deg,#1e293b,#020617);
    border: 1px solid #334155;
    border-radius: 10px;
    color: white;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #0ea5e9;
    color: black;
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
# SIDEBAR
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Predictor"

col1, col2 = st.sidebar.columns(2)

if col1.button("🔮 Predictor"):
    st.session_state.page = "Predictor"

if col2.button("📊 Analytics"):
    st.session_state.page = "Analytics"

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
st.caption(f"Model Accuracy (R²): {round(score,2)}")

# =====================================================
# PREDICTOR
# =====================================================

if page == "Predictor":

    st.subheader("📍 Select City")
    city = st.selectbox("City", sorted(data["City"].unique()))

    st.subheader("🧪 Enter Pollution Levels")

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
# ANALYTICS
# =====================================================

elif page == "Analytics":

    st.subheader("👤 Human Impact Analyzer")

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