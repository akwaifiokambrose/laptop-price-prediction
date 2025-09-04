import streamlit as st
import pandas as pd
import numpy as np
# Note: joblib is used by scikit-learn's load function
import joblib

# Set page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Title and description
st.title("💻 Laptop Price Predictor")
st.markdown("""
This app predicts the price of a laptop based on its specifications.
Use the sliders and selectors below to enter your laptop details.
""")

# Load the trained model
model_loaded = False
model = None
try:
    # Load the model (you'll need to save it first)
    model = joblib.load('laptop_price_model.pkl')
    model_loaded = True
    # st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error loading model: Model file 'laptop_price_model.pkl' not found. Please ensure it exists in the repository.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Input features
if model_loaded:
    st.header("Enter Laptop Specifications")

    # --- Collect inputs for all 17 features ---

    # Feature 1: laptop_ID (Assumed not critical for prediction, set to 0)
    laptop_id = 0 # Placeholder or user input if needed

    # Feature 2: Inches (Screen Size)
    inches = st.slider("Screen Size (Inches)", 10.0, 20.0, 15.0)

    # Feature 3: Company_encoded
    company_options = {
        "Apple": 4, "Asus": 5, "Dell": 0, "HP": 1, "Acer": 3,
        "Lenovo": 2, "MSI": 8, "Microsoft": 6, "Toshiba": 7, "Huawei": 9,
        "Xiaomi": 10, "Vero": 11, "Chuwi": 12, "Google": 13, "Fujitsu": 14,
        "LG": 15, "Samsung": 16
    }
    company_selected = st.selectbox("Company", list(company_options.keys()))
    company_encoded = company_options[company_selected]

    # Feature 4: Product_encoded (Simplified - use Company or a general category)
    # Since Product is highly specific, we can map a simplified version or use Company_encoded again
    # Or ask for a general product line if you have mappings
    # For simplicity here, we'll use a placeholder or map based on company
    # You might want to refine this mapping based on your original data
    product_mapping = {
        "Apple": {"MacBook Air": 40, "MacBook Pro": 41, "MacBook": 42}, # Example
        "Dell": {"Inspiron": 10, "XPS": 11, "Alienware": 12}, # Example
        "HP": {"Pavilion": 20, "EliteBook": 21, "ProBook": 22, "Spectre": 23}, # Example
        "Lenovo": {"ThinkPad": 30, "Yoga": 31, "IdeaPad": 32}, # Example
        # Add more mappings as needed or use a default
    }
    # Get possible products for the selected company
    possible_products = product_mapping.get(company_selected, {"General": 999})
    product_selected = st.selectbox("Product Line (Simplified)", list(possible_products.keys()))
    product_encoded = possible_products[product_selected]

    # Feature 5: ScreenResolution_encoded
    resolution_options = {
        "1366x768": 0, "1600x900": 1, "1920x1080": 2, "2160x1440": 3,
        "2256x1504": 4, "2304x1440": 5, "2496x1664": 6, "2560x1440": 7,
        "2560x1600": 8, "2736x1824": 9, "2880x1800": 10, "3200x1800": 11,
        "3840x2160": 12
    }
    resolution_selected = st.selectbox("Screen Resolution", list(resolution_options.keys()))
    screenresolution_encoded = resolution_options[resolution_selected]

    # Feature 6: Cpu_encoded
    cpu_options = {
        "Intel Core i3": 10, "Intel Core i5": 11, "Intel Core i7": 12,
        "Intel Core i9": 13, "AMD Ryzen 3": 20, "AMD Ryzen 5": 21,
        "AMD Ryzen 7": 22, "AMD Ryzen 9": 23, "Other Intel": 19,
        "Other AMD": 29, "ARM": 30
    }
    cpu_selected = st.selectbox("CPU", list(cpu_options.keys()))
    cpu_encoded = cpu_options[cpu_selected]

    # Feature 7: Memory_encoded
    # Assuming mapping like 8GB=0, 16GB=1, etc. You need to check your original encoding.
    # Let's assume RAM GB slider maps to an encoded value.
    ram_gb = st.slider("RAM (GB)", 2, 64, 8)
    # Example simple mapping (you need to use your actual mapping)
    # Let's assume 2GB=0, 4GB=1, 8GB=2, 16GB=3, 32GB=4, 64GB=5
    ram_mapping = {2: 0, 4: 1, 8: 2, 16: 3, 32: 4, 64: 5}
    memory_encoded = ram_mapping.get(ram_gb, 2) # Default to 8GB encoding if not found

    # Feature 8: Gpu_encoded
    gpu_options = {
        "Intel HD Graphics": 0, "Intel UHD Graphics": 1,
        "NVIDIA GeForce GTX": 10, "NVIDIA GeForce RTX": 11,
        "AMD Radeon": 20, "AMD Radeon Vega": 21,
        "NVIDIA Quadro": 12, "NVIDIA MX": 13
    }
    gpu_selected = st.selectbox("GPU", list(gpu_options.keys()))
    gpu_encoded = gpu_options[gpu_selected]

    # Feature 9: Weight_encoded
    # This might be the encoded version of weight category or a direct encoding.
    # If it's based on Weight_KG, we can derive it or ask for a category.
    # Let's ask for a category for simplicity.
    weight_cat_options = {"Ultra Light (<1.5kg)": 0, "Light (1.5-2kg)": 1, "Medium (2-2.5kg)": 2, "Heavy (>2.5kg)": 3}
    weight_cat_selected = st.selectbox("Weight Category", list(weight_cat_options.keys()))
    weight_encoded = weight_cat_options[weight_cat_selected]

    # Feature 10: Screen_Size_Inches (Same as Inches)
    screen_size_inches = inches

    # Feature 11: Weight_KG
    weight_kg = st.slider("Weight (KG)", 0.5, 5.0, 1.5)

    # Feature 12: Brand_Popularity (Example scale)
    brand_popularity = st.slider("Brand Popularity (1-100)", 1, 100, 50)

    # Feature 13: Has_SSD
    storage_type = st.selectbox("Storage Type", ["HDD Only", "SSD Only", "SSD + HDD"])
    has_ssd = 1 if "SSD" in storage_type else 0

    # Feature 14: Has_HDD
    has_hdd = 1 if "HDD" in storage_type else 0

    # Feature 15: CPU_Brand_Score (0=Other, 1=AMD, 2=Intel)
    cpu_brand_score_map = {"Other Intel": 2, "Intel Core i3": 2, "Intel Core i5": 2, "Intel Core i7": 2, "Intel Core i9": 2,
                           "Other AMD": 1, "AMD Ryzen 3": 1, "AMD Ryzen 5": 1, "AMD Ryzen 7": 1, "AMD Ryzen 9": 1,
                           "ARM": 0}
    cpu_brand_score = cpu_brand_score_map.get(cpu_selected, 0)

    # Feature 16: GPU_Brand_Score (0=Integrated/Other, 1=AMD, 2=NVIDIA)
    gpu_brand_score_map = {"Intel HD Graphics": 0, "Intel UHD Graphics": 0,
                           "NVIDIA GeForce GTX": 2, "NVIDIA GeForce RTX": 2,
                           "NVIDIA Quadro": 2, "NVIDIA MX": 2,
                           "AMD Radeon": 1, "AMD Radeon Vega": 1}
    gpu_brand_score = gpu_brand_score_map.get(gpu_selected, 0)

    # Feature 17: Performance_Score (Example calculation)
    performance_score = cpu_brand_score + gpu_brand_score + (ram_gb / 4.0)


    # Predict button
    if st.button("Predict Price"):
        # Prepare input data IN THE EXACT ORDER THE MODEL EXPECTS
        # Order: ['laptop_ID', 'Inches', 'Company_encoded', 'Product_encoded', 'ScreenResolution_encoded',
        #         'Cpu_encoded', 'Memory_encoded', 'Gpu_encoded', 'Weight_encoded', 'Screen_Size_Inches',
        #         'Weight_KG', 'Brand_Popularity', 'Has_SSD', 'Has_HDD', 'CPU_Brand_Score',
        #         'GPU_Brand_Score', 'Performance_Score']
        input_data = np.array([[
            laptop_id, inches, company_encoded, product_encoded, screenresolution_encoded,
            cpu_encoded, memory_encoded, gpu_encoded, weight_encoded, screen_size_inches,
            weight_kg, brand_popularity, has_ssd, has_hdd, cpu_brand_score,
            gpu_brand_score, performance_score
        ]])

        try:
            # Make prediction
            predicted_price = model.predict(input_data)[0]
            # Display result
            st.success(f"💡 Predicted Price: €{predicted_price:.2f}")
        except ValueError as e:
            st.error(f"Error making prediction: {e}. Please check the input data format.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

else:
    st.warning("⚠️ Model not loaded. Prediction cannot be performed.")
    
