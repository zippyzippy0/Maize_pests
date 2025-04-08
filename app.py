# app.py
import streamlit as st
import joblib
import numpy as np
import requests
from streamlit_lottie import st_lottie
import pydeck as pdk

# Load models and encoders
model_fall = joblib.load("model_fall_armyworm.pkl")
model_ear = joblib.load("model_ear_rot.pkl")
model_stem = joblib.load("model_stem_borer.pkl")

le_fall = joblib.load("encoder_fall.pkl")
le_ear = joblib.load("encoder_ear.pkl")
le_stem = joblib.load("encoder_stem.pkl")

API_KEY = "8169413357cba4f829589924f1b1742c"

# Page setup
st.set_page_config(
    page_title="🌽 Maize Pest Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style
st.markdown("""
    <style>
        :root {
            color-scheme: light dark;
        }
        html, body {
            transition: background-color 0.5s ease;
        }
        .prediction-box {
            border-radius: 1rem;
            padding: 1.2rem;
            margin-top: 1rem;
            background-color: rgba(0, 0, 0, 0.05);
            transition: all 0.4s ease-in-out;
            animation: fadein 0.8s ease-in;
        }
        .stButton>button {
            transition: all 0.3s ease;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #4CAF50;
            color: white;
            transform: scale(1.03);
        }
        @keyframes fadein {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Weather fetch
@st.cache_data
def get_weather_by_city(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return {
            "city": data["name"],
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"],
            "temp_max": data["main"]["temp_max"],
            "temp_min": data["main"]["temp_min"],
            "rainfall": data.get("rain", {}).get("1h", 0.0),
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    return None

# Prediction logic
def predict_pests(temp_max, temp_min, rainfall, humidity, wind_speed):
    features = np.array([[temp_max, temp_min, rainfall, humidity, wind_speed]])
    fall = le_fall.inverse_transform([model_fall.predict(features)[0]])[0]
    ear = le_ear.inverse_transform([model_ear.predict(features)[0]])[0]
    stem = le_stem.inverse_transform([model_stem.predict(features)[0]])[0]
    return fall, ear, stem

# App title
st.title("🌽 Maize Pest Invasion Predictor")
st.write("Enter current weather data or use live data to estimate pest threat levels in your maize farm.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌐 Choose weather input method:")
    method = st.radio("", ["Manual Entry", "Enter City"])
    weather = {}

    if method == "Enter City":
        city = st.text_input("🏙️ Enter City Name (e.g., Nairobi)", "Nyeri")
        if city:
            weather = get_weather_by_city(city)
            if weather:
                st.success(f"✅ Weather for {weather['city']}")
                st.write(f"**🌡️ Temp Max:** {weather['temp_max']}°C")
                st.write(f"**🌡️ Temp Min:** {weather['temp_min']}°C")
                st.write(f"**🌧️ Rainfall:** {weather['rainfall']} mm")
                st.write(f"**💧 Humidity:** {weather['humidity']}%")
                st.write(f"**💨 Wind Speed:** {weather['wind_speed']} m/s")
            else:
                st.warning("⚠️ Could not fetch data. Check city name.")
    else:
        weather["temp_max"] = st.number_input("🌡️ Max Temperature (°C)", 10.0, 50.0, 33.0)
        weather["temp_min"] = st.number_input("🌡️ Min Temperature (°C)", 5.0, 40.0, 20.0)
        weather["rainfall"] = st.number_input("🌧️ Rainfall (mm)", 0.0, 20.0, 0.5)
        weather["humidity"] = st.number_input("💧 Humidity (%)", 10.0, 100.0, 60.0)
        weather["wind_speed"] = st.number_input("💨 Wind Speed (m/s)", 0.1, 10.0, 2.0)

    if st.button("🔍 Predict", use_container_width=True):
        fall, ear, stem = predict_pests(
            weather["temp_max"],
            weather["temp_min"],
            weather["rainfall"],
            weather["humidity"],
            weather["wind_speed"]
        )
        with col_right:
            st.subheader("🧾 Prediction Results")
            st.markdown(f"<div class='prediction-box'>🪲 <strong>Fall Armyworm:</strong> {fall}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-box'>🌽 <strong>Ear Rot:</strong> {ear}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-box'>🕷️ <strong>Stem Borer:</strong> {stem}</div>", unsafe_allow_html=True)

            if fall == "High" or stem == "High":
                st.markdown("""### 🔔 Early Warning System
                Farmers and agronomists are advised to:
                - Apply preventive measures if risk is Moderate or High.
                - Monitor maize fields closely for pest symptoms.
                - Use weather data regularly to track potential pest outbreaks.""")

# 📍 Map Visualization
if method == "Enter City" and weather.get("lat") and weather.get("lon"):
    st.subheader("🗺️ Maize Pest Risk Map")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v12",
        initial_view_state=pdk.ViewState(
            latitude=weather["lat"],
            longitude=weather["lon"],
            zoom=6,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=[{
                    "position": [weather["lon"], weather["lat"]],
                    "size": 200,
                }],
                get_position="position",
                get_radius=20000,
                get_color=[255, 0, 0],
                pickable=True,
            )
        ],
    ))
