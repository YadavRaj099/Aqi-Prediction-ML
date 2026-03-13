🌍 AI Air Quality Index Predictor

A machine learning web application that predicts Air Quality Index (AQI) using environmental pollutant data and visualizes pollution trends across Indian cities.

This project demonstrates how machine learning and data visualization can be used to analyze and predict air pollution levels.

---

🚀 Live Application

Access the deployed application here:

Streamlit App:
https://aqi-prediction-ml-fgvblan5oquucprnaezqmp.streamlit.app/

---

📌 Project Overview

Air pollution is a major environmental concern affecting millions of people.
This project uses historical pollution data to:

- Predict AQI using machine learning
- Visualize pollution distribution across Indian cities
- Analyze pollution trends over time
- Rank cities based on pollution levels

The application is designed as an interactive environmental analytics dashboard.

---

🧠 Machine Learning Model

The AQI prediction model is trained using historical pollutant measurements.

Model Used

Random Forest Regressor

Features Used

Feature
PM2.5
PM10
NO
NO2
NOx
NH3
CO
SO2
O3
Benzene
Toluene
Xylene

These pollutants are commonly used in AQI calculation and air quality monitoring systems.

---

🧰 Technologies Used

Python
Streamlit
Scikit-learn
Pandas
NumPy
Plotly

These technologies were used for:

- Machine learning model training
- interactive web application development
- data visualization
- environmental data analysis

---

🌍 Application Features

1️⃣ AQI Prediction Tool

A wizard-style interface where users enter pollutant concentrations to estimate AQI.

2️⃣ India Pollution Heatmap

A geographic visualization showing AQI distribution across major Indian cities.

3️⃣ Pollution Trend Analysis

Interactive graphs that display pollutant trends over time.

4️⃣ City Pollution Leaderboard

Ranks cities by their average AQI levels.

5️⃣ Health Advisory

Based on predicted AQI levels, the app provides simple health recommendations.

---

📊 AQI Classification

AQI Range| Air Quality
0 – 50| Good
51 – 100| Satisfactory
101 – 200| Moderate
201 – 300| Poor
301 – 400| Very Poor
401 – 500| Severe

---

📂 Dataset

Dataset used in this project:

city_day.csv

It contains air quality measurements from multiple Indian cities including pollutants such as:

- PM2.5
- PM10
- NO₂
- CO
- SO₂
- O₃

---

⚙️ Installation

Clone the repository

git clone https://github.com/YadavRaj099/Aqi-Prediction-ML
cd Aqi-Prediction-ML

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run aqi_app.py

---

📸 Example Output

The application provides:

- AQI prediction
- pollution heatmap of India
- pollutant trend graphs
- city pollution rankings

---

🎯 Project Goal

The goal of this project is to demonstrate how machine learning and data visualization can help monitor environmental conditions and make pollution data easier to understand.

---

⚠️ Disclaimer

This application is an educational machine learning project and should not be used for official environmental monitoring or health decisions.

---

👨‍💻 Author

Raj Yadav

GitHub:
https://github.com/YadavRaj099

---

⭐ Support

If you found this project useful, consider giving it a star on GitHub.