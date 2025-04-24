import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- CONFIG --------------------
st.set_page_config(page_title="🌽 Maize Pest Predictor", layout="wide")

model = joblib.load("pest_severity_model.pkl")
feature_names = joblib.load("feature_names.pkl")
severity_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
API_KEY = "091fd5c1ab03ae28846c1748ea358f97"  # Replace with your own OpenWeatherMap API key

# -------------------- WEATHER FETCH --------------------
@st.cache_data
def get_weather(city):
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
                "wind_speed": data["wind"]["speed"],
                "weather": data["weather"][0]["description"].capitalize()
            }
    except:
        return None
    return None

# -------------------- REGION MAPPING BY COORDINATES --------------------
def get_region_from_coordinates(lat, lon):
    if lat >= -4 and lat <= 4 and lon >= 33 and lon <= 42:
        if lat < 0:
            if lon > 38:
                return "Coastal"
            elif lon < 36:
                return "Western"
            else:
                return "Nyanza"
        elif lat >= 0 and lon < 37:
            return "Rift Valley"
        elif lat >= 0 and lon >= 37 and lat < 1:
            return "Central"
        elif lat >= 1:
            return "Eastern"
    return "ASAL areas"

# -------------------- UI --------------------
st.markdown("## 🌾 AI Maize Pest Severity Predictor")
st.markdown("Predict pest severity using season, region, and crop condition — powered by AI and real-time weather.")

with st.form("prediction_form"):
    city = st.text_input("📍 Enter your county/town/city", "Nyeri")
    season = st.selectbox("📅 Season", ["Jan–March", "Feb–April", "March–May", "Aug–Oct", "Sept–Dec", "Oct–Nov"])
    crop_stage = st.selectbox("🌱 Crop Stage", ["Seedling", "Vegetative", "Tasseling", "Silking", "Grain filling", "Maturity"])
    pest = st.selectbox("🦟 Pest", ["Fall Armyworm", "Corn Earworm", "Locust"])
    condition = st.text_area("🧪 Field Condition (Trigger)", "Dry spells followed by rainfall")
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    weather = get_weather(city)
    if weather:
        region = get_region_from_coordinates(weather["lat"], weather["lon"])

        # Prepare input
        input_data = {
            f"Pest_{pest}": 1,
            f"Season/Month_{season}": 1,
            f"Crop Stage Affected_{crop_stage}": 1,
            f"Location/Region_{region}": 1
        }

        # TF-IDF (pre-fitted with known phrases)
        tfidf = TfidfVectorizer(max_features=25, stop_words='english')
        tfidf.fit(["Dry spells followed by rainfall", "Warm dry spells", "Heavy rains", "Cross-border swarm movement"])
        tfidf_input = tfidf.transform([condition])
        tfidf_cols = [f"TFIDF_{w}" for w in tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_input.toarray(), columns=tfidf_cols)

        # Combine all features
        full_input = {col: input_data.get(col, 0) for col in feature_names}
        for col in tfidf_df.columns:
            if col in full_input:
                full_input[col] = tfidf_df[col].values[0]

        input_df = pd.DataFrame([full_input])
        prediction = model.predict(input_df)[0]
        severity = severity_map.get(prediction, "Unknown")

        # -------------------- LAYOUT --------------------
        top_left, top_right = st.columns(2)

        with top_left:
            st.markdown("### 📍 Location & Crop Info")
            st.markdown(f"- **City:** {city.title()}")
            st.markdown(f"- **Season:** {season}")
            st.markdown(f"- **Crop Stage:** {crop_stage}")
            st.markdown(f"- **Region (AI-detected):** {region}")

        with top_right:
            st.markdown("### 🦟 Pest Info")
            st.markdown(f"- **Pest:** {pest}")
            st.markdown(f"- **Trigger:** {condition if condition.strip() else 'N/A'}")

        mid_left, mid_right = st.columns(2)

        with mid_left:
            st.subheader(f"🌦️ Weather in {city.title()}")
            weather_table = pd.DataFrame({
                "Weather": [weather["weather"]],
                "Temp (°C)": [f'{weather["temp_min"]} – {weather["temp_max"]}'],
                "Humidity (%)": [weather["humidity"]],
                "Rainfall (mm)": [weather["rainfall"]],
                "Wind (m/s)": [weather["wind_speed"]]
            })
            st.table(weather_table)

        with mid_right:
            st.subheader("📊 Prediction Summary")
            result_df = pd.DataFrame({
                "Pest": [pest],
                "Severity": [severity],
                "Crop Stage": [crop_stage],
                "Season": [season],
                "Region": [region]
            })
            st.table(result_df)

            if severity == "Very High":
                st.error("⚠️ Very High risk! Immediate action required.")
            elif severity == "High":
                st.warning("🔶 High risk. Monitor and apply IPM (Integrated Pest Management) measures.")
            elif severity == "Medium":
                st.info("🟡 Medium risk. Preventive action advised.")
            else:
                st.success("🟢 Low risk. Continue good agronomic practices.")

        # Bottom map
        st.markdown("---")
        st.subheader("🗺️ Location Map")
        st.map(pd.DataFrame([{"lat": weather["lat"], "lon": weather["lon"]}]))
    else:
        st.warning("🌐 Could not fetch weather. Check city spelling or internet connection.")
