import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI AQI Predictor", layout="wide")

# -----------------------------
# CUSTOM UI STYLE
# -----------------------------

st.markdown("""
<style>

.stApp{
background-image:url("https://images.unsplash.com/photo-1509395176047-4a66953fd231");
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

[data-testid="stMetricValue"]{
font-size:28px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.title("🌍 AI Air Quality Predictor")

st.caption(
"Predict Air Quality Index using machine learning and pollutant levels."
)

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

        pm25 = st.number_input(
        "PM2.5",0.0,1000.0,50.0)

        pm10 = st.number_input(
        "PM10",0.0,1000.0,80.0)

        no2 = st.number_input(
        "NO2",0.0,500.0,20.0)

    with col2:

        so2 = st.number_input(
        "SO2",0.0,500.0,10.0)

        co = st.number_input(
        "CO",0.0,10.0,0.8)

        o3 = st.number_input(
        "O3",0.0,500.0,30.0)

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
        color="green"

    elif prediction <= 100:
        category="Satisfactory"
        color="yellow"

    elif prediction <= 200:
        category="Moderate"
        color="orange"

    elif prediction <= 300:
        category="Poor"
        color="red"

    elif prediction <= 400:
        category="Very Poor"
        color="purple"

    else:
        category="Severe"
        color="maroon"

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
        st.warning(
        "Sensitive individuals should limit outdoor activity."
        )

    else:
        st.error(
        "Avoid prolonged outdoor exposure."
        )

    if st.button("Restart"):
        st.session_state.step=1
        st.rerun()

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.caption(
"AI AQI Predictor • Environmental Machine Learning Project"
)