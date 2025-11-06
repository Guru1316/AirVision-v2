# 🌫️ AirVision - AI-Powered Air Quality Intelligence Dashboard

A real-time air quality monitoring and forecasting platform designed for **Delhi-NCR**, combining **machine learning**, **live environmental data**, **health advisory**, and **policy planning support**.

🔗 **Live Dashboard:** [https://air-vision.streamlit.app/](https://air-vision.streamlit.app/)

---

## 🚀 Key Features

* **Live AQI Monitoring** powered by WAQI API
* **72-Hour AQI Forecast** using SARIMAX time-series modeling
* **Pollution Source Attribution** using Machine Learning feature interpretation
* **Health-Based Advisory System** (mask suggestion, outdoor safety guidance)
* **GIS Interactive Map** of Delhi-NCR pollution hotspots
* **Policy Strategy Simulation** (What-if intervention impact calculator)
* **Responsive UI** built with Streamlit & custom CSS

---

## 🧠 System Architecture

```
Data Sources → Preprocessing → ML/AQI Forecasting → Streamlit Dashboard → Advisory & Maps
```

* **Random Forest Model** for pollutant contribution inference
* **SARIMAX Model** for forecasting AQI
* **Folium + OpenStreetMap** for real-time mapping

---

## 📊 Tech Stack

| Component          | Technology                  |
| ------------------ | --------------------------- |
| Frontend UI        | Streamlit + Custom CSS      |
| AQI Forecast Model | SARIMAX (Statsmodels)       |
| Source Detection   | RandomForest (Scikit-learn) |
| Data Fetch         | WAQI API                    |
| Mapping            | Folium + streamlit-folium   |

---

## 📂 Project Structure

```
AirVision/
│
├── Models/
│   ├── aqi_model.pkl               # RandomForest model
│   ├── delhi_aqi_forecast_sarimax.pkl   # SARIMAX forecast model
│
├── app.py                          # Streamlit main application
├── requirements.txt
└── README.md
```

---

## 🌍 Live Data Usage

AirVision fetches real-time AQI and station data from:

> [https://aqicn.org/api/](https://aqicn.org/api/)

To use your own token:

```
set WAQI_TOKEN=your_token_here
```

---

## 📦 Installation

```
git clone https://github.com/yourusername/AirVision.git
cd AirVision
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧭 Pages Overview

| Page                | Description                                |
| ------------------- | ------------------------------------------ |
| 🌍 Overview         | Live AQI + health advisory + contributions |
| 📈 Forecast         | 72-hour forecast + severity alerts         |
| 🧭 Sources & Policy | Source breakdown + policy simulation       |
| 🗺️ Live Map        | Real-time pollution hotspot mapping        |
| ℹ️ About            | System design & methodology                |

---

## 🩺 Health Advisory Scale

| AQI Range | Category        | Recommendation           |
| --------- | --------------- | ------------------------ |
| 0-50      | Good 🌱         | No mask needed           |
| 51-100    | Satisfactory 🙂 | Mask optional            |
| 101-200   | Moderate 😐     | Light mask advised       |
| 201-300   | Poor 😷         | N95 recommended          |
| 301-400   | Very Poor 😵    | Avoid outdoor activities |
| 401+      | Severe ☠️       | Stay indoors strictly    |

---

## ⚡ Future Enhancements

* 🔄 Daily & seasonal AQI pattern learning
* 🛰️ Integration of satellite fire & dust plume data (MODIS / VIIRS)
* 📱 Mobile app interface
* 🏛️ Government policy analytics panel

---

## 🧑‍💻 Developer

**Guruprasad K**
*Dedicated to Clean & Green India* 🇮🇳

---

## ⭐ Support

If you found this project useful:

```
⭐ Star this repository  
```

Together, let's build a cleaner future 🌍✨
