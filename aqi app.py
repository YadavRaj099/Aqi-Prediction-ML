import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

st.info("Model Used: Random Forest Regressor • Dataset: City AQI Data")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

with st.spinner("Loading data..."):
    data = load_data()

st.success(f"📍 Available Cities: {data['City'].nunique()}")

# =====================================================
# MODEL (REAL TRAIN/TEST SPLIT)
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

with st.spinner("Training model..."):
    model, score = train_model(data)

st.success(f"Model Accuracy (R² Score): {round(score, 2)}")

# =====================================================
# HOW IT WORKS
# =====================================================

st.markdown("### ⚙️ How it Works")
st.write("""
- Uses Random Forest Regressor  
- Trained on historical AQI dataset  
- Predicts AQI based on pollutant levels  
- Provides health advisory based on predicted AQI  
""")

# =====================================================
# IMPACT
# =====================================================

st.markdown("### 🌍 Impact")
st.write("""
This system helps individuals and authorities monitor air quality 
and take preventive actions to reduce health risks.
""")

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(["🔮 Predictor", "📊 Analytics"])

# =====================================================
# PREDICTOR
# =====================================================

with tab1:

    if "step" not in st.session_state:
        st.session_state.step = 1

    st.progress(st.session_state.step / 3)

    if st.session_state.step == 1:

        st.subheader("📍 Select Location")

        city = st.selectbox("City", sorted(data["City"].unique()))
        st.session_state.city = city

        # Latest AQI
        latest = data[data["City"] == city].sort_values("Date").iloc[-1]
        st.metric("Latest AQI", round(latest["AQI"], 2))

        if st.button("Next →", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    elif st.session_state.step == 2:

        st.subheader("🧪 Enter Pollution Levels")

        # AUTO-FILL BUTTON
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

        col1, col2 = st.columns(2)

        defaults = st.session_state.get("inputs", [50,80,20,10,0.8,30])

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

        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

        with col2:
            if st.button("🚀 Predict AQI", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

    elif st.session_state.step == 3:

        st.subheader("📊 AQI Prediction")

        pm25, pm10, no2, so2, co, o3 = st.session_state.inputs

        input_data = np.array([[
            pm25, pm10, 10, no2, 20, 5,
            co, so2, o3, 1, 2, 1
        ]])

        with st.spinner("Predicting AQI..."):
            try:
                prediction = round(model.predict(input_data)[0], 2)

                st.success("Prediction generated successfully!")

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

                st.subheader("Health Advisory")

                if prediction < 100:
                    st.success("Air quality is acceptable.")
                elif prediction < 200:
                    st.warning("Sensitive individuals should limit outdoor activity.")
                else:
                    st.error("Avoid prolonged exposure.")

            except:
                st.error("Prediction failed.")

        if st.button("Restart", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# =====================================================
# ANALYTICS
# =====================================================

with tab2:

    st.subheader("India AQI Pollution Heatmap")

    city_avg = data.groupby("City")["AQI"].mean().reset_index()

    st.dataframe(city_avg)

    st.subheader("Pollution Trend Analysis")

    pollutant = st.selectbox("Select pollutant",
        ["PM2.5","PM10","NO2","SO2","CO","O3"])

    trend = data.groupby("Date")[pollutant].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most Polluted Cities")

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