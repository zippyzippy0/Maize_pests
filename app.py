import streamlit as st
import joblib
import numpy as np
import requests
import pandas as pd
import geocoder

# Load models and encoders
model_fall = joblib.load("model_fall_armyworm.pkl")
model_ear = joblib.load("model_ear_rot.pkl")
model_stem = joblib.load("model_stem_borer.pkl")

le_fall = joblib.load("encoder_fall.pkl")
le_ear = joblib.load("encoder_ear.pkl")
le_stem = joblib.load("encoder_stem.pkl")

API_KEY = "8169413357cba4f829589924f1b1742c"

st.set_page_config(page_title="🌽 Maize Pest Predictor", layout="wide")

# Styling
st.markdown("""
    <style>
    :root { color-scheme: light dark; }
    .prediction-box {
        border-radius: 1rem;
        padding: 1rem;
        background-color: var(--secondary-background-color);
        transition: all 0.4s ease-in-out;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Weather fetcher
@st.cache_data
def get_weather_by_city(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return [
            data['main']['temp_max'],
            data['main']['temp_min'],
            data['rain'].get('1h', 0.0) if 'rain' in data else 0.0,
            data['main']['humidity'],
            data['wind']['speed']
        ]
    return None

# Prediction function
def predict_pests(temp_max, temp_min, rainfall, humidity, wind_speed):
    features = np.array([[temp_max, temp_min, rainfall, humidity, wind_speed]])
    pred_fall = model_fall.predict(features)[0]
    pred_ear = model_ear.predict(features)[0]
    pred_stem = model_stem.predict(features)[0]
    return (
        le_fall.inverse_transform([pred_fall])[0],
        le_ear.inverse_transform([pred_ear])[0],
        le_stem.inverse_transform([pred_stem])[0]
    )

# Title and instructions
st.title("🌽 Maize Pest Invasion Predictor")
st.write("Enter current weather data or use live data to estimate pest threat levels in your maize farm.")

# Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Choose weather input method:")
    method = st.radio("", ["Manual Entry", "Enter City"])
    
    weather_data = None

    if method == "Enter City":
        city = st.text_input("🏙️ Enter City Name (e.g., Nairobi)", "Nyeri")
        if st.button("🌦️ Fetch Weather"):
            weather_data = get_weather_by_city(city)
            if weather_data:
                st.session_state['weather_data'] = weather_data
            else:
                st.warning("⚠️ Could not fetch weather data.")
    else:
        temp_max = st.number_input("🌡️ Max Temperature (°C)", 10.0, 50.0, 33.0)
        temp_min = st.number_input("🌡️ Min Temperature (°C)", 5.0, 40.0, 20.0)
        rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 100.0, 0.5)
        humidity = st.number_input("💧 Humidity (%)", 10.0, 100.0, 60.0)
        wind_speed = st.number_input("💨 Wind Speed (m/s)", 0.1, 20.0, 2.0)

        weather_data = [temp_max, temp_min, rainfall, humidity, wind_speed]

    if st.button("🔍 Predict"):
        if method == "Enter City" and 'weather_data' not in st.session_state:
            st.warning("⚠️ Please fetch weather data first.")
        else:
            if method == "Enter City":
                temp_max, temp_min, rainfall, humidity, wind_speed = st.session_state['weather_data']
            fall, ear, stem = predict_pests(temp_max, temp_min, rainfall, humidity, wind_speed)

            with col2:
                st.subheader("🧾 Prediction Results")
                st.markdown(f"<div class='prediction-box'>🪲 <strong>Fall Armyworm:</strong> {fall}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-box'>🌽 <strong>Ear Rot:</strong> {ear}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-box'>🕷️ <strong>Stem Borer:</strong> {stem}</div>", unsafe_allow_html=True)

                st.markdown("### ☁️ Weather Conditions Used:")
                st.write(f"**🌡️ Max Temp:** {temp_max} °C")
                st.write(f"**🌡️ Min Temp:** {temp_min} °C")
                st.write(f"**🌧️ Rainfall:** {rainfall} mm")
                st.write(f"**💧 Humidity:** {humidity} %")
                st.write(f"**💨 Wind Speed:** {wind_speed} m/s")

                if fall == "High" or stem == "High":
                    st.markdown("""
                        ### 🔔 Early Warning System
                        Farmers are advised to:
                        - Apply preventive pest measures.
                        - Monitor maize fields regularly.
                        - Use this tool frequently to stay informed.
                    """)

# Map Visualization
st.subheader("🗺️ Maize Pest Risk Map (Beta)")

try:
    g = geocoder.ip("me")
    df = pd.DataFrame([[g.latlng[0], g.latlng[1]]], columns=['lat', 'lon'])
    st.map(df)
except:
    st.warning("Could not detect location for map.")
