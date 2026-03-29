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

.card {
    background: linear-gradient(180deg,#0f172a,#020617);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #1e293b;
    margin-bottom: 20px;
}

h1, h2, h3 { color: white; }

.subtitle { color: #94a3b8; }

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
    transition: 0.3s;
}

div.stButton > button:hover {
    background: #0ea5e9;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR (BUTTON NAVIGATION)
# =====================================================

st.sidebar.markdown("## 🌍 AI AQI App")
st.sidebar.markdown("---")

if "page" not in st.session_state:
    st.session_state.page = "Predictor"

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔮 Predictor", use_container_width=True):
        st.session_state.page = "Predictor"

with col2:
    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.page = "Analytics"

st.sidebar.markdown("---")

if st.session_state.page == "Predictor":
    st.sidebar.success("Currently: 🔮 Predictor")
else:
    st.sidebar.success("Currently: 📊 Analytics")

st.sidebar.markdown("---")
st.sidebar.markdown("✨ AI Powered AQI System")

page = st.session_state.page

# =====================================================
# HERO
# =====================================================

st.markdown("""
<h1>🌍 AI Air Quality Predictor</h1>
<p class="subtitle">
Predict Air Quality Index using machine learning and pollutant levels.
</p>
""", unsafe_allow_html=True)

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

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, score

model, score = train_model(data)

# =====================================================
# INFO BOX
# =====================================================

st.markdown(f"""
<div style="
    background: linear-gradient(135deg,#1e293b,#020617);
    padding:20px;
    border-radius:12px;
    border:1px solid #334155;
    margin-bottom:20px;
">
    <h4 style="color:#38bdf8;">⚙️ Model Information</h4>
    <p style="color:#cbd5f5;">
    • Model: Random Forest Regressor<br>
    • Accuracy (R²): {round(score,2)}<br>
    • Cities Covered: {data['City'].nunique()}
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PREDICTOR
# =====================================================

if page == "Predictor":

    if "step" not in st.session_state:
        st.session_state.step = 1

    st.progress(st.session_state.step / 3)

    if st.session_state.step == 1:

        st.subheader("📍 Select Location")

        city = st.selectbox("City", sorted(data["City"].unique()))
        st.session_state.city = city

        latest = data[data["City"] == city].sort_values("Date").iloc[-1]
        st.metric("Latest AQI", round(latest["AQI"], 2))

        if st.button("Next →", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    elif st.session_state.step == 2:

        st.subheader("🧪 Enter Pollution Levels")

        if st.button("Use Real Data"):
            city_data = data[data["City"] == st.session_state.city].iloc[-1]
            st.session_state.inputs = [
                city_data["PM2.5"],
                city_data["PM10"],
                city_data["NO2"],
                city_data["SO2"],
                city_data["CO"],
                city_data["O3"]
            ]

        defaults = st.session_state.get("inputs", [50,80,20,10,0.8,30])

        col1, col2 = st.columns(2)

        with col1:
            pm25 = st.number_input("PM2.5", 0.0, 1000.0, float(defaults[0]))
            pm10 = st.number_input("PM10", 0.0, 1000.0, float(defaults[1]))
            no2 = st.number_input("NO2", 0.0, 500.0, float(defaults[2]))

        with col2:
            so2 = st.number_input("SO2", 0.0, 500.0, float(defaults[3]))
            co = st.number_input("CO", 0.0, 10.0, float(defaults[4]))
            o3 = st.number_input("O3", 0.0, 500.0, float(defaults[5]))

        st.session_state.inputs = [pm25, pm10, no2, so2, co, o3]

        col1, col2 = st.columns(2)

        if col1.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

        if col2.button("🚀 Predict AQI", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    elif st.session_state.step == 3:

        st.subheader("📊 AQI Prediction")

        pm25, pm10, no2, so2, co, o3 = st.session_state.inputs

        input_data = np.array([[
            pm25, pm10, 10, no2, 20, 5,
            co, so2, o3, 1, 2, 1
        ]])

        prediction = round(model.predict(input_data)[0], 2)

        if prediction <= 50:
            category, color = "Good", "#00e400"
        elif prediction <= 100:
            category, color = "Satisfactory", "#ffff00"
        elif prediction <= 200:
            category, color = "Moderate", "#ff7e00"
        elif prediction <= 300:
            category, color = "Poor", "#ff0000"
        elif prediction <= 400:
            category, color = "Very Poor", "#8f3f97"
        else:
            category, color = "Severe", "#7e0023"

        col1, col2 = st.columns(2)

        col1.metric("AQI", prediction)
        col2.markdown(
            f"<div class='badge' style='background:{color};color:black'>{category}</div>",
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            gauge={'axis': {'range': [0, 500]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Restart", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# =====================================================
# ANALYTICS
# =====================================================

elif page == "Analytics":

    st.subheader("🌍 India AQI Heatmap")

    city_avg = data.groupby("City")["AQI"].mean().reset_index()

    city_coords = {
        "Delhi": (28.61,77.20),
        "Mumbai": (19.07,72.87),
        "Ahmedabad": (23.02,72.57),
        "Bangalore": (12.97,77.59),
        "Chennai": (13.08,80.27),
        "Kolkata": (22.57,88.36),
        "Hyderabad": (17.38,78.48),
        "Pune": (18.52,73.85)
    }

    city_avg["lat"] = city_avg["City"].map(lambda x: city_coords.get(x,(None,None))[0])
    city_avg["lon"] = city_avg["City"].map(lambda x: city_coords.get(x,(None,None))[1])
    city_avg = city_avg.dropna()

    fig = px.scatter_geo(
        city_avg,
        lat="lat",
        lon="lon",
        size="AQI",
        color="AQI",
        hover_name="City",
        color_continuous_scale="Turbo"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Trend")

    pollutant = st.selectbox("Pollutant",
        ["PM2.5","PM10","NO2","SO2","CO","O3"])

    trend = data.groupby("Date")[pollutant].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Most Polluted Cities")

    leaderboard = (
        data.groupby("City")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(leaderboard, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("Built with Machine Learning • AQI Prediction System")