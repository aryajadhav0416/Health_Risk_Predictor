# Health Risk Prediction Web App with Explainability

A machine learning project that predicts an individual’s health risk (low or high) based on lifestyle factors using a synthetic dataset. The project includes training multiple classification models, analyzing feature importance with SHAP explainability, and deploying an interactive Streamlit web application for real-time prediction and interpretation.

---

## 🚀 Features

- **Multi-model training and evaluation:** Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine, and XGBoost classifiers.
- **Data preprocessing:** Handles mixed numeric and categorical data with label encoding.
- **Explainable AI:** Utilizes SHAP (SHapley Additive exPlanations) to provide both global and local interpretability of model predictions.
- **Interactive Streamlit app:** Input lifestyle data and get instant health risk predictions with visual explanations of feature impacts.
- **Model and explainer serialization:** Saves trained models, encoders, and SHAP explainers for efficient reuse and deployment.

---

## 📊 Dataset

The dataset contains synthetic data simulating real-world health-related factors:

| Feature       | Description                             | Type                 | Example          |
|---------------|-------------------------------------|---------------------|------------------|
| age           | Age in years                         | Numeric             | 35               |
| weight        | Weight in kilograms                  | Numeric             | 70               |
| height        | Height in centimeters                | Numeric             | 172              |
| exercise      | Exercise frequency level             | Categorical         | medium           |
| sleep         | Average sleep hours per night        | Numeric             | 7                |
| sugar_intake  | Sugar consumption level              | Categorical         | high             |
| smoking       | Smoking habit                       | Categorical         | no               |
| alcohol       | Alcohol consumption habit            | Categorical         | yes              |
| married       | Marital status                      | Categorical         | yes              |
| profession    | Type of profession                   | Categorical         | teacher          |
| bmi           | Body Mass Index (calculated)         | Numeric             | 24.5             |
| health_risk   | Target label (low/high health risk)  | Categorical (target) | high             |

---

## 📋 Getting Started

### Prerequisites

- Python 3.7+
- Pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/health-risk-prediction.git
   cd health-risk-prediction
