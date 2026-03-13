import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI AQI Predictor", layout="wide")

# -----------------------------
# BACKGROUND STYLE
# -----------------------------

st.markdown("""
<style>

.stApp{
background-image:url("https://images.unsplash.com/photo-1581092580497-e0d23cbdf1dc");
background-size:cover;
background-attachment:fixed;
}

.block-container{
background:rgba(10,15,25,0.85);
padding:40px;
border-radius:15px;
}

h1,h2,h3{
color:white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.title("🌍 AI Air Quality Predictor")
st.caption("Predict Air Quality Index using machine learning and pollutant levels.")

st.info(
"This tool predicts AQI using environmental pollutant inputs. Educational use only."
)

# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("city_day.csv")
    df = df.dropna()

    return df

data = load_data()

# -----------------------------
# MODEL TRAINING
# -----------------------------

features = [
"PM2.5","PM10","NO","NO2","NOx",
"NH3","CO","SO2","O3","Benzene",
"Toluene","Xylene"
]

X = data[features]
y = data["AQI"]

model = RandomForestRegressor()
model.fit(X,y)

# -----------------------------
# SESSION STATE
# -----------------------------

if "step" not in st.session_state:
    st.session_state.step = 1

step = st.session_state.step

st.progress(step/3)

# -----------------------------
# STEP 1
# -----------------------------

if step == 1:

    st.subheader("Step 1 • Location")

    city = st.selectbox(
    "Select City",
    data["City"].unique()
    )

    st.session_state.city = city

    if st.button("Next"):
        st.session_state.step = 2
        st.rerun()

# -----------------------------
# STEP 2
# -----------------------------

elif step == 2:

    st.subheader("Step 2 • Pollution Inputs")

    col1,col2 = st.columns(2)

    with col1:

        pm25 = st.number_input("PM2.5",0.0,1000.0,50.0)
        pm10 = st.number_input("PM10",0.0,1000.0,80.0)
        no2 = st.number_input("NO2",0.0,500.0,20.0)

    with col2:

        so2 = st.number_input("SO2",0.0,500.0,10.0)
        co = st.number_input("CO",0.0,10.0,0.8)
        o3 = st.number_input("O3",0.0,500.0,30.0)

    st.session_state.pm25 = pm25
    st.session_state.pm10 = pm10
    st.session_state.no2 = no2
    st.session_state.so2 = so2
    st.session_state.co = co
    st.session_state.o3 = o3

    col1,col2 = st.columns(2)

    with col1:
        if st.button("Previous"):
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("Predict AQI"):
            st.session_state.step = 3
            st.rerun()

# -----------------------------
# STEP 3
# -----------------------------

elif step == 3:

    st.subheader("AQI Prediction Result")

    pm25 = st.session_state.pm25
    pm10 = st.session_state.pm10
    no2 = st.session_state.no2
    so2 = st.session_state.so2
    co = st.session_state.co
    o3 = st.session_state.o3

# assumed values

    no = 10
    nox = 20
    nh3 = 5
    benzene = 1
    toluene = 2
    xylene = 1

# prediction

    input_data = np.array([[
    pm25,pm10,no,no2,nox,nh3,
    co,so2,o3,benzene,toluene,xylene
    ]])

    prediction = model.predict(input_data)[0]
    prediction = round(prediction,2)

# category

    if prediction <= 50:
        category="Good"
    elif prediction <= 100:
        category="Satisfactory"
    elif prediction <= 200:
        category="Moderate"
    elif prediction <= 300:
        category="Poor"
    elif prediction <= 400:
        category="Very Poor"
    else:
        category="Severe"

# score cards

    col1,col2 = st.columns(2)

    col1.metric("Predicted AQI",prediction)
    col2.metric("Air Quality",category)

# gauge meter

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text':"AQI Level"},
        gauge={
        'axis':{'range':[0,500]},
        'steps':[
        {'range':[0,50],'color':"#00e400"},
        {'range':[50,100],'color':"#ffff00"},
        {'range':[100,200],'color':"#ff7e00"},
        {'range':[200,300],'color':"#ff0000"},
        {'range':[300,400],'color':"#8f3f97"},
        {'range':[400,500],'color':"#7e0023"}
        ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

# health advisory

    st.subheader("Health Advisory")

    if prediction < 100:
        st.success("Air quality is acceptable.")
    elif prediction < 200:
        st.warning("Sensitive individuals should limit outdoor activity.")
    else:
        st.error("Avoid prolonged outdoor exposure.")

    if st.button("Restart"):
        st.session_state.step=1
        st.rerun()

# -----------------------------
# REAL INDIA AQI MAP
# -----------------------------

st.markdown("---")
st.subheader("India AQI Pollution Heatmap")

# city coordinates

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

lats = []
lons = []
aqi_vals = []
cities = []

for _,row in city_avg.iterrows():

    city = row["City"]

    if city in city_coords:

        lat,lon = city_coords[city]

        lats.append(lat)
        lons.append(lon)
        aqi_vals.append(row["AQI"])
        cities.append(city)

fig_map = go.Figure(go.Scattergeo(

    lat = lats,
    lon = lons,
    text = cities,
    marker=dict(
        size=12,
        color=aqi_vals,
        colorscale="Reds",
        colorbar_title="AQI",
        line=dict(width=1,color="black")
    )

))

fig_map.update_layout(

    title="Air Pollution Distribution Across Indian Cities",

    geo=dict(
        scope="asia",
        showcountries=True,
        showland=True,
        landcolor="rgb(217,217,217)"
    )

)

st.plotly_chart(fig_map,use_container_width=True)

# -----------------------------
# POLLUTION TREND GRAPH
# -----------------------------

st.markdown("---")
st.subheader("Pollution Trend Analysis")

pollutant = st.selectbox(
"Select pollutant",
["PM2.5","PM10","NO2","SO2","CO","O3"]
)

trend = data.groupby("Date")[pollutant].mean()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=trend.index,
    y=trend.values,
    mode="lines"
))

fig.update_layout(
    title=f"{pollutant} Trend Over Time",
    xaxis_title="Date",
    yaxis_title="Concentration"
)

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# CITY POLLUTION LEADERBOARD
# -----------------------------

st.markdown("---")
st.subheader("Most Polluted Cities")

city_rank = data.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)

leaderboard = city_rank.reset_index()

st.dataframe(
leaderboard,
use_container_width=True
)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")
st.caption("AI AQI Predictor • Environmental Machine Learning Project")