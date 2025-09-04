import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Title and description
st.title("💻 Laptop Price Predictor")
st.markdown("""
This app predicts the price of a laptop based on its specifications.
Use the sliders below to enter your laptop details.
""")

# Load the trained model
try:
    # Load the model (you'll need to save it first)
    model = joblib.load('laptop_price_model.pkl')
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Input features
if model_loaded:
    st.header("Enter Laptop Specifications")

    # RAM (GB)
    ram = st.slider("RAM (GB)", 2, 64, 8)

    # Screen Size (Inches)
    screen_size = st.slider("Screen Size (Inches)", 10, 20, 15)

    # Weight (KG)
    weight = st.slider("Weight (KG)", 0.5, 5.0, 1.5)

    # Brand Popularity (based on frequency)
    brand_popularity = st.slider("Brand Popularity", 1, 100, 50)

    # CPU Brand Score (0=Other, 1=AMD, 2=Intel)
    cpu_brand = st.selectbox("CPU Brand", ["Other", "AMD", "Intel"])
    cpu_score = {"Other": 0, "AMD": 1, "Intel": 2}[cpu_brand]

    # GPU Brand Score (0=Integrated, 1=AMD, 2=NVIDIA)
    gpu_brand = st.selectbox("GPU Brand", ["Integrated", "AMD", "NVIDIA"])
    gpu_score = {"Integrated": 0, "AMD": 1, "NVIDIA": 2}[gpu_brand]

    # Storage Type
    storage_type = st.selectbox("Storage Type", ["HDD Only", "SSD Only", "SSD + HDD"])
    storage_score = {"HDD Only": 0, "SSD Only": 1, "SSD + HDD": 2}[storage_type]

    # Performance Score
    performance_score = cpu_score + gpu_score + ram / 4

    # Predict button
    if st.button("Predict Price"):
        # Prepare input data
        input_data = np.array([[ram, screen_size, weight, brand_popularity, \
                               cpu_score, gpu_score, storage_score, performance_score]])

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        # Display result
        st.success(f"💡 Predicted Price: €{predicted_price:.2f}")

else:
    st.warning("⚠️ Model not loaded. Please ensure 'laptop_price_model.pkl' exists.")