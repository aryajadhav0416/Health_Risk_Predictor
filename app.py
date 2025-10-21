import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Load saved model, encoders, target encoder, and shap explainer
model = joblib.load('health_risk_model.pkl')
encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
explainer = joblib.load('shap_explainer.pkl')

st.set_page_config(page_title="Health Risk Predictor", layout="centered")
st.title("🩺 Health Risk Prediction App")
st.markdown("Enter your lifestyle info to predict your health risk.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100, 30)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        height = st.number_input("Height (cm)", 100, 250, 170)
        sleep = st.slider("Sleep Hours (per night)", 3.0, 12.0, 7.0)
        married = st.selectbox("Married?", ['yes', 'no'])

    with col2:
        exercise = st.selectbox("Exercise Level", ['none', 'low', 'medium', 'high'])
        sugar_intake = st.selectbox("Sugar Intake", ['low', 'medium', 'high'])
        smoking = st.selectbox("Do you smoke?", ['yes', 'no'])
        alcohol = st.selectbox("Alcohol Consumption?", ['yes', 'no'])
        profession = st.selectbox("Profession", ['office_worker', 'teacher', 'doctor', 'engineer'])

    submitted = st.form_submit_button("Predict Health Risk")

if submitted:
    bmi = weight / ((height / 100) ** 2)

    input_df = pd.DataFrame([{
        'age': age,
        'weight': weight,
        'height': height,
        'exercise': encoders['exercise'].transform([exercise])[0],
        'sleep': sleep,
        'sugar_intake': encoders['sugar_intake'].transform([sugar_intake])[0],
        'smoking': encoders['smoking'].transform([smoking])[0],
        'alcohol': encoders['alcohol'].transform([alcohol])[0],
        'married': encoders['married'].transform([married])[0],
        'profession': encoders['profession'].transform([profession])[0],
        'bmi': bmi
    }])

    pred_encoded = model.predict(input_df)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    st.markdown("---")
    st.success(f"🏥 **Predicted Health Risk:** `{pred_label.upper()}`")
    st.metric("Calculated BMI", f"{bmi:.2f}")

    # Show SHAP summary plot in sidebar
    st.sidebar.header("Model Explainability")
    if st.sidebar.checkbox("Show SHAP Summary Plot"):
        image = Image.open('shap_summary_plot.png')
        st.sidebar.image(image, caption="SHAP Summary Plot - Feature Impact")

    # Local SHAP explanation (force/waterfall plot)
    st.subheader("🔍 Feature Impact for Your Prediction")

    shap_values = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig, bbox_inches='tight')