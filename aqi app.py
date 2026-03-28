import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="AQI Predictor", layout="wide")

# =====================================================
# STYLES (UPGRADED UI)
# =====================================================

st.markdown("""
<style>

body {
    background: #020617;
}

.block-container {
    padding: 2rem;
}

/* Card UI */
.card {
    background: linear-gradient(180deg,#0f172a,#020617);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #1e293b;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

/* Headings */
h1, h2, h3 {
    color: white;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

/* Badge */
.badge {
    padding: 12px 18px;
    border-radius: 12px;
    font-weight: bold;
    display: inline-block;
    font-size: 16px;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
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

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

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

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

with st.spinner("Training model..."):
    model = train_model(data)

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(["🔮 Predictor", "📊 Analytics"])

# =====================================================
# PREDICTOR TAB
# =====================================================

with tab1:

    if "step" not in st.session_state:
        st.session_state.step = 1

    st.progress(st.session_state.step / 3)

    # ---------------- STEP 1 ----------------
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

    # ---------------- STEP 2 ----------------
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

    # ---------------- STEP 3 ----------------
    elif st.session_state.step == 3:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("📊 AQI Prediction")

        pm25, pm10, no2, so2, co, o3 = st.session_state.inputs

        input_data = np.array([[
            pm25, pm10, 10, no2, 20, 5,
            co, so2, o3, 1, 2, 1
        ]])

        try:
            prediction = round(model.predict(input_data)[0], 2)
            st.success("Prediction generated successfully!")

            # CATEGORY
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

            # GAUGE (UPGRADED)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                gauge={
                    'axis': {'range': [0, 500]},
                    'steps': [
                        {'range': [0, 50], 'color': "#00e400"},
                        {'range': [50, 100], 'color': "#ffff00"},
                        {'range': [100, 200], 'color': "#ff7e00"},
                        {'range': [200, 300], 'color': "#ff0000"},
                        {'range': [300, 400], 'color': "#8f3f97"},
                        {'range': [400, 500], 'color': "#7e0023"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # ADVISORY
            st.subheader("Health Advisory")

            if prediction < 100:
                st.success("Air quality is acceptable. Enjoy your day!")
            elif prediction < 200:
                st.warning("Sensitive individuals should limit outdoor activity.")
            else:
                st.error("Avoid prolonged exposure. Consider wearing a mask.")

        except:
            st.error("Prediction failed. Please check your inputs.")

        if st.button("Restart", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ANALYTICS TAB
# =====================================================

with tab2:

    st.subheader("India AQI Pollution Heatmap")

    city_coords = {
        "Delhi": (28.6139,77.2090),
        "Mumbai": (19.0760,72.8777),
        "Ahmedabad": (23.0225,72.5714),
        "Bangalore": (12.9716,77.5946),
        "Chennai": (13.0827,80.2707),
        "Kolkata": (22.5726,88.3639),
        "Hyderabad": (17.3850,78.4867),
        "Pune": (18.5204,73.8567),
        "Jaipur": (26.9124,75.7873),
        "Lucknow": (26.8467,80.9462),
        "Chandigarh": (30.7333,76.7794),
        "Bhopal": (23.2599,77.4126),
        "Patna": (25.5941,85.1376),
        "Amaravati": (16.5062,80.6480)
    }

    city_avg = data.groupby("City")["AQI"].mean().reset_index()

    lats, lons, aqi_vals, cities = [], [], [], []

    for _, row in city_avg.iterrows():
        if row["City"] in city_coords:
            lat, lon = city_coords[row["City"]]
            lats.append(lat)
            lons.append(lon)
            aqi_vals.append(row["AQI"])
            cities.append(row["City"])

    fig_map = go.Figure(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=cities,
        marker=dict(
            size=12,
            color=aqi_vals,
            colorscale="Reds",
            colorbar_title="AQI"
        )
    ))

    fig_map.update_layout(geo=dict(scope="asia"))

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Pollution Trend Analysis")

    pollutant = st.selectbox(
        "Select pollutant",
        ["PM2.5","PM10","NO2","SO2","CO","O3"]
    )

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