import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Simple demo
st.title("🌾 Crop Yield Predictor (Fixed!)")
st.write("This proves sklearn is installed correctly!")

# Test prediction with dummy data
X = np.array([[1], [2], [3]])
y = np.array([10, 20, 30])
model = RandomForestRegressor()
model.fit(X, y)
prediction = model.predict([[4]])[0]

st.success(f"Test prediction for input '4': {prediction:.1f} (If you see this, it works!)")
