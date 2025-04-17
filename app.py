import streamlit as st
import numpy as np
import joblib
import requests
import pydeck as pdk
import os

# Optional: Set public Mapbox token
os.environ["MAPBOX_API_KEY"] = "pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBnd3Z6eWl2eWgifQ.-0lBXRQ0tNMa1l8HxI1ERA"

# Load all pest models
pest_cols = ['fall_armyworm', 'ear_rot', 'stem_borer', 'corn_earworm', 'locust']
models = {pest: joblib.load(f"saved_models/{pest}_model.pkl") for pest in pest_cols}

# Set page config
st.set_page_config(page_title="🌽 Maize Pest Predictor", layout="wide")
st.title("🌾 Maize Pest and Disease Invasion Predictor")
st.write("Use weather data to predict pest invasion levels and help farmers make informed pest control decisions.")

API_KEY = "8169413357cba4f829589924f1b1742c"

@st.cache_data
def get_weather_by_city(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
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
    except:
        pass
    return None

def predict_pest_risks(data):
    input_features = np.array([[
        data["temp_max"], data["temp_min"], data["rainfall"],
        data["humidity"], data["wind_speed"],
        data["soil_moisture"], data["ndvi"], data["altitude"]
    ]])

    severity_map = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
    results = {pest.replace("_", " ").title(): severity_map[models[pest].predict(input_features)[0]] for pest in pest_cols}
    return results

# UI
col1, col2 = st.columns(2)
with col1:
    method = st.radio("Select weather input method", ["Manual Entry", "Enter City"])
    weather = {}

    if method == "Enter City":
        city = st.text_input("Enter city name", "Nyeri")
        if city:
            weather = get_weather_by_city(city)
            if weather:
                st.success(f"Weather data for {weather['city']}")
                st.write(weather)
            else:
                st.error("City not found or API error.")
    else:
        weather["temp_max"] = st.number_input("🌡️ Max Temperature (°C)", 10.0, 50.0, 33.0)
        weather["temp_min"] = st.number_input("🌡️ Min Temperature (°C)", 5.0, 40.0, 20.0)
        weather["rainfall"] = st.number_input("🌧️ Rainfall (mm)", 0.0, 20.0, 0.5)
        weather["humidity"] = st.number_input("💧 Humidity (%)", 10.0, 100.0, 60.0)
        weather["wind_speed"] = st.number_input("💨 Wind Speed (m/s)", 0.1, 10.0, 2.0)

    weather["soil_moisture"] = st.number_input("🌱 Soil Moisture (%)", 10.0, 50.0, 25.0)
    weather["ndvi"] = st.number_input("🌿 NDVI (0.1–0.9)", 0.1, 0.9, 0.5)
    weather["altitude"] = st.number_input("⛰️ Altitude (m)", 100.0, 2500.0, 1500.0)

    if st.button("🔍 Predict", use_container_width=True):
        preds = predict_pest_risks(weather)
        with col2:
            st.subheader("🧾 Prediction Results")
            for pest, risk in preds.items():
                st.markdown(f"<div style='padding:10px; background:#f0f0f0; border-radius:10px; margin:5px 0;'>🔍 <strong>{pest}:</strong> {risk}</div>", unsafe_allow_html=True)

            if "High" in preds.values():
                st.markdown("""
                    ### 🚨 High Risk Detected
                    Farmers should:
                    - Apply **preventive treatment**
                    - **Scout** regularly
                    - Use traps and monitor frequently
                """)

# Show map if weather has location info
if method == "Enter City" and weather.get("lat") and weather.get("lon"):
    st.subheader("🗺️ Maize Pest Risk Map")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=weather["lat"],
            longitude=weather["lon"],
            zoom=6,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"position": [weather["lon"], weather["lat"]], "size": 200}],
                get_position="position",
                get_radius=20000,
                get_color=[255, 0, 0],
                pickable=True,
            )
        ]
    ))
