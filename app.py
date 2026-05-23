import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Prediction Portal",
    page_icon="📊",
    layout="wide"
)

# --- LOAD MODEL ARTIFACT ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_churn_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model_artifact = load_model()

# --- HEADER SECTION ---
st.title("📊 Customer Churn Prediction Portal")
st.markdown("""
This diagnostic application uses an optimized **XGBoost Machine Learning model** to predict the likelihood of customer churn. 
Adjust the customer profile parameters below to simulate retention risks and view real-time probability updates.
""")
st.divider()

if model_artifact is None:
    st.error("⚠️ **Model Artifact Missing:** Please verify that `models/xgboost_churn_model.pkl` exists before running the interface.")
    st.stop()

model = model_artifact['model']
expected_features = model_artifact['features']

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Customer Profile Settings")

# 1. Core Demographics
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

st.sidebar.divider()

# 2. Account & Usage Metrics
tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

st.sidebar.divider()

# 3. Services Subscribed
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# Dependent Services (Only active if Internet is enabled)
if internet_service != "No":
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
else:
    online_security = "No internet service"
    tech_support = "No internet service"
    streaming_movies = "No internet service"

st.sidebar.divider()

# 4. Financial Charges
monthly_charges = st.sidebar.number_input("Monthly Bill ($)", min_value=15.0, max_value=120.0, value=65.0)
# Auto-estimate total charges for simulation simplicity if user doesn't specify
total_charges = tenure * monthly_charges

# --- DATA PREPROCESSING FUNCTION ---
def prepare_input_data():
    """Maps UI input elements to the expected mathematical schema."""
    # Build foundational dictionary matching raw CSV format
    input_dict = {
        'gender': 1 if gender == "Female" else 0, # LabelEncoder mapping: Female=1, Male=0
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'StreamingMovies': streaming_movies
    }
    
    # Convert to single-row DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Apply dummy encoding for categorical features
    multi_cols = ['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingMovies']
    df_encoded = pd.get_dummies(df_input, columns=multi_cols, drop_first=False)
    
    # Align structural schema to match model's expected features exactly
    final_features = pd.DataFrame(0, index=[0], columns=expected_features)
    
    for col in df_encoded.columns:
        if col in final_features.columns:
            final_features[col] = df_encoded[col]
            
    # Force boolean columns to floats to maintain platform compatibility
    return final_features.astype(float)

# --- EXECUTE PREDICTION ---
input_features = prepare_input_data()
churn_probability = model.predict_proba(input_features)[0][1]
churn_prediction = model.predict(input_features)[0]

# --- MAIN DASHBOARD DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Diagnostic Results")
    
    # Visual Output Cards
    if churn_prediction == 1:
        st.error(f"⚠️ **High Churn Risk Detected**")
        st.metric(label="Calculated Probability of Leaving", value=f"{churn_probability * 100:.1f}%")
        st.markdown("""
        **Operational Recommendation:** This account exhibits strong withdrawal indicators. Immediately trigger proactive customer retention workflows (e.g., promotional discounts, contract extension incentives, or direct outreach from customer success).
        """)
    else:
        st.success(f"✅ **Account Appears Stable**")
        st.metric(label="Calculated Probability of Leaving", value=f"{churn_probability * 100:.1f}%")
        st.markdown("""
        **Operational Recommendation:** This profile matches historical retention patterns. Continue standard lifecycle management without intervention.
        """)

with col2:
    st.subheader("Active Feature Matrix")
    st.caption("Raw parameter state passed to the XGBoost calculation framework:")
    # Display non-zero input parameters for diagnostic transparency
    display_df = input_features.loc[:, (input_features != 0).any(axis=0)]
    st.dataframe(display_df.T, height=250)

# --- FOOTER ---
st.divider()
st.caption("System Architecture: End-to-end Data Science Portfolio Demonstration using Scikit-Learn, XGBoost, and Streamlit.")