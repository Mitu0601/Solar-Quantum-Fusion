#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# ==============================================
# 🌊 Solar Radiation Predictor - Teal Green Dashboard
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Solar Radiation Predictor",
    page_icon="☀️",
    layout="wide"
)

# -------------------------------
# Custom Teal/Green Dark CSS Styling + Animations
# -------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #0e1e1e;
    color: #e0f7f1;
    font-family: 'Segoe UI', Roboto, sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #104d4d !important;
    border-right: 1px solid #0c3a3a;
}

/* Headers */
h1, h2, h3, h4 {
    color: #00bfa5;
    text-shadow: 0 0 10px rgba(0,191,165,0.3);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00bfa5, #1de9b6);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 4px 10px rgba(0,191,165,0.4);
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 6px 15px rgba(29,233,182,0.6);
}

/* Prediction Box */
.pred-box {
    background: linear-gradient(145deg, #014d4d, #0e2e2e);
    border-left: 6px solid #00bfa5;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 0 20px rgba(0,191,165,0.2);
    animation: fadeIn 1s ease;
}
.pred-value {
    animation: pulse 1.5s infinite;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes pulse {
    0% {text-shadow: 0 0 10px #00bfa5;}
    50% {text-shadow: 0 0 25px #1de9b6;}
    100% {text-shadow: 0 0 10px #00bfa5;}
}

/* Cards for Home */
.feature-card {
    background: #014d4d;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,191,165,0.2);
    margin: 10px;
}
.feature-card h3 {
    color: #1de9b6;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Define Preprocessor
# -------------------------------
class SolarPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['DateTime'] = pd.to_datetime(X['Data'].astype(str) + ' ' + X['Time'])
        X['Hour'] = X['DateTime'].dt.hour
        X['Month'] = X['DateTime'].dt.month
        X['DayOfYear'] = X['DateTime'].dt.dayofyear
        X['Sunrise'] = pd.to_datetime(X['TimeSunRise'])
        X['Sunset'] = pd.to_datetime(X['TimeSunSet'])
        X['DayLength'] = (X['Sunset'] - X['Sunrise']).dt.seconds / 3600
        for col in ['Speed','Temperature','Pressure']:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        X['Day_sin'] = np.sin(2*np.pi*X['DayOfYear']/365)
        X['Day_cos'] = np.cos(2*np.pi*X['DayOfYear']/365)
        X['WindDirection(Degrees)_sin'] = np.sin(2*np.pi*X['WindDirection(Degrees)']/360)
        X['WindDirection(Degrees)_cos'] = np.cos(2*np.pi*X['WindDirection(Degrees)']/360)
        X['Hour_sin'] = np.sin(2*np.pi*X['Hour']/24)
        X['Hour_cos'] = np.cos(2*np.pi*X['Hour']/24)
        X['Month_sin'] = np.sin(2*np.pi*X['Month']/12)
        X['Month_cos'] = np.cos(2*np.pi*X['Month']/12)
        drop_cols = ['UNIXTime','Data','Time','TimeSunRise','TimeSunSet','DateTime','Sunrise','Sunset','Hour','Month','DayOfYear','WindDirection(Degrees)']
        X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True)
        return X

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open("solar_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("Model file 'solar_pipeline.pkl' not found.")
        return None
pipeline = load_model()

# -------------------------------
# Session State for History
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# -------------------------------
# Navigation Tabs
# -------------------------------
tabs = st.tabs(["🏠 Home", "☀️ Prediction", "📊 Insights", "📝 Project Summary"])

# ============================================
# 🏠 Home Page
# ============================================
with tabs[0]:
    st.markdown("<h1 style='text-align:center;'>☀️ Solar Radiation Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#1de9b6;'>Interactive dashboard to predict solar radiation using environmental parameters and time features.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Feature cards grid
    col1, col2, col3 = st.columns(3)
    col1.markdown("<div class='feature-card'><h3>🌡️ Inputs</h3><p>Temperature, Pressure, Humidity, Wind & Time</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='feature-card'><h3>☀️ Prediction</h3><p>Predict solar radiation using RandomForest + preprocessor pipeline</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='feature-card'><h3>📊 Insights</h3><p>Feature importance visualization and prediction trends</p></div>", unsafe_allow_html=True)

# ============================================
# ☀️ Prediction Page
# ============================================
with tabs[1]:
    st.header("☀️ Enter Weather Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Date")
        time = st.time_input("Time")
    with col2:
        sunrise = st.time_input("Sunrise Time")
        sunset = st.time_input("Sunset Time")
    with col3:
        temp = st.slider("🌡️ Temperature (°F)", 0, 120, 75)
        pressure = st.slider("🌬️ Pressure (hPa)", 900, 1100, 1012)

    col4, col5 = st.columns(2)
    with col4:
        humidity = st.slider("💧 Humidity (%)", 0, 100, 45)
    with col5:
        speed = st.slider("💨 Wind Speed (m/s)", 0, 50, 5)
        wind_dir = st.slider("🧭 Wind Direction (°)", 0, 360, 180)

    input_df = pd.DataFrame({
        'Data':[date.strftime("%Y-%m-%d")],
        'Time':[time.strftime("%H:%M:%S")],
        'TimeSunRise':[sunrise.strftime("%H:%M:%S")],
        'TimeSunSet':[sunset.strftime("%H:%M:%S")],
        'Temperature':[temp],
        'Pressure':[pressure],
        'Humidity':[humidity],
        'Speed':[speed],
        'WindDirection(Degrees)':[wind_dir]
    })

    if st.button("🚀 Predict Solar Radiation", use_container_width=True):
        if pipeline is not None:
            try:
                predicted_log = pipeline.predict(input_df)[0]
                predicted = np.expm1(predicted_log)

                # Input Report Card
                st.markdown("<h4>📝 Entered Parameters</h4>", unsafe_allow_html=True)
                st.table(input_df.T.rename(columns={0:"Value"}))

                # Prediction Box
                st.markdown(
                    f"""
                    <div class='pred-box'>
                        <p style='font-size:1.2em;margin:0;'>Predicted Solar Radiation</p>
                        <p class='pred-value' style='font-size:2.5em;font-weight:bold;margin:8px 0;'>{predicted:.2f}</p>
                        <p style='font-size:1.1em;'>W/m²</p>
                    </div>
                    """, unsafe_allow_html=True
                )

                # Add to history
                result_row = input_df.copy()
                result_row["PredictedRadiation"] = predicted
                st.session_state["history"] = pd.concat([st.session_state["history"], result_row], ignore_index=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.warning("Pipeline not loaded.")

    # CSV Download
    if not st.session_state["history"].empty:
        csv = st.session_state["history"].to_csv(index=False).encode('utf-8')
        st.download_button("📂 Download Prediction History", data=csv, file_name="prediction_history.csv")

# ============================================
# 📊 Insights Page
# ============================================
with tabs[2]:
    st.header("📊 Feature Importance")
    if st.session_state["history"].empty:
        st.info("No predictions yet.")
    else:
        latest = st.session_state["history"].iloc[-1]
        # Weight-based importance
        feature_importance = {
            "Temperature": latest["Temperature"]*0.25,
            "Pressure": latest["Pressure"]*0.05,
            "Humidity": latest["Humidity"]*0.15,
            "Speed": latest["Speed"]*0.2,
            "Wind Direction": latest["WindDirection(Degrees)"]*0.1,
            "Day Length": ((pd.to_datetime(latest["TimeSunSet"]) - pd.to_datetime(latest["TimeSunRise"])).seconds/3600)*0.25
        }

        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(list(feature_importance.keys()), list(feature_importance.values()), color="#00bfa5")
        ax.set_xlabel("Relative Impact (pseudo)")
        ax.set_title("Feature Contribution")
        st.pyplot(fig)

# ============================================
# 📝 Project Summary Page
# ============================================
with tabs[3]:
    st.header("📝 Project Summary")
    st.markdown("""
    ### Project Overview
    This dashboard predicts **solar radiation** using environmental parameters and time-based features.

    ### Model Used
    - **A hybrid deep learning model-CNN,Bilstm,Attention,Transformer and XGBoost Residual Learner** integrated in a pipeline with preprocessing

    ### Preprocessing Steps
    - Combined date & time features
    - Calculated **Day Length**, **Sin/Cos transforms** for cyclical features (Hour, Month, Day of Year, Wind Direction)
    - Outlier handling for Temperature, Pressure, Speed

    ### Features Included
    - Temperature, Pressure, Humidity, Wind Speed & Direction, Day Length, Hour, Month, DayOfYear

    ### Outputs
    - Predicted solar radiation (W/m²)
    - Feature importance visualization
    

    """)


# In[ ]:




