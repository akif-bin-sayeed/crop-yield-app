import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Verify installations (hidden in final app)
st.write(f"Running with numpy=={np.__version__}, pandas=={pd.__version__}")  # Remove this line after verification

# Title
st.title("🌾 Smart Crop Yield Predictor")
st.write("Predict yields based on crop type, pests, and environment")

# Dummy data generation
@st.cache_resource
def load_model():
    np.random.seed(42)
    data = {
        'NDVI': np.random.uniform(0.2, 0.8, 200),
        'Rainfall_mm': np.random.randint(500, 1200, 200),
        'Temperature_C': np.random.uniform(20, 35, 200),
        'Soil_pH': np.random.uniform(5.5, 7.5, 200),
        'Fertilizer_kg_ha': np.random.randint(50, 200, 200),
        'Pest_Incidence': np.random.choice(['None', 'Low', 'Medium', 'High'], 200),
        'Crop_Type': np.random.choice(['Rice', 'Wheat', 'Maize', 'Soybean'], 200),
        'Yield_ton_ha': np.random.uniform(2.5, 6.0, 200)
    }
    df = pd.DataFrame(data)
    
    # Convert categories to numbers
    df['Pest_Incidence'] = df['Pest_Incidence'].map({'None':0, 'Low':1, 'Medium':2, 'High':3})
    df['Crop_Type'] = df['Crop_Type'].map({'Rice':0, 'Wheat':1, 'Maize':2, 'Soybean':3})
    
    # Train model
    X = df.drop('Yield_ton_ha', axis=1)
    y = df['Yield_ton_ha']
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

model = load_model()

# User inputs
col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("Crop Type", ['Rice', 'Wheat', 'Maize', 'Soybean'])
    ndvi = st.slider("NDVI (Plant Health)", 0.2, 0.8, 0.5)
    rainfall = st.slider("Rainfall (mm)", 500, 1200, 800)
with col2:
    pest = st.selectbox("Pest Level", ['None', 'Low', 'Medium', 'High'])
    temp = st.slider("Temperature (°C)", 20, 35, 25)
    fertilizer = st.slider("Fertilizer (kg/ha)", 50, 200, 100)

# Prediction
if st.button("Predict Yield"):
    # Map inputs
    pest_num = {'None':0, 'Low':1, 'Medium':2, 'High':3}[pest]
    crop_num = {'Rice':0, 'Wheat':1, 'Maize':2, 'Soybean':3}[crop]
    
    input_data = pd.DataFrame([{
        'NDVI': ndvi,
        'Rainfall_mm': rainfall,
        'Temperature_C': temp,
        'Soil_pH': 6.5,  # Default value
        'Fertilizer_kg_ha': fertilizer,
        'Pest_Incidence': pest_num,
        'Crop_Type': crop_num
    }])
    
    prediction = model.predict(input_data)[0]
    st.success(f"**Predicted Yield:** {prediction:.2f} tons/ha")
    st.balloons()

# Sample data table
st.subheader("Sample Data Reference")
st.table(pd.DataFrame({
    'Crop': ['Rice', 'Wheat', 'Maize', 'Soybean'],
    'Pest Impact': ['-10%', '-5%', '-20%', '-15%'],
    'Avg Yield': ['4.2 tons/ha', '3.8 tons/ha', '5.1 tons/ha', '4.5 tons/ha']
}))
