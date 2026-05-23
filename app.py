import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Customer Diagnostics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM ENTERPRISE THEME INJECTION ---
# This overrides the default Streamlit engine to guarantee dark console aesthetics
st.markdown("""
    <style>
    /* Force dark theme consistency across the entire viewport */
    .stApp {
        background-color: #0E1117 !important;
        color: #E2E8F0 !important;
    }
    
    /* Style Sidebar container explicitly to prevent color splitting */
    [data-testid="stSidebar"] {
        background-color: #0B0D13 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Ensure all sidebar text overrides to high contrast white/silver */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #F8FAFC !important;
    }
    
    /* Custom formatting for the main application title */
    .main-title {
        color: #F8FAFC !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.05rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    .sub-title {
        color: #94A3B8 !important;
        font-size: 1.05rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Clean custom cards for business protocols */
    .protocol-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }
    .protocol-critical {
        background-color: rgba(198, 40, 40, 0.15) !important;
        border-color: #C62828 !important;
        color: #FCA5A5 !important;
    }
    .protocol-warning {
        background-color: rgba(249, 168, 37, 0.1) !important;
        border-color: #F9A825 !important;
        color: #FDE047 !important;
    }
    .protocol-stable {
        background-color: rgba(46, 125, 50, 0.15) !important;
        border-color: #2E7D32 !important;
        color: #86EFAC !important;
    }

    /* Remove residual default workspace margins */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem !important; padding-bottom: 0rem !important;}
    
    /* Custom styling for dataframes */
    .stDataFrame {
        border: 1px solid #1E293B !important;
        border-radius: 0.375rem !important;
        overflow: hidden;
    }

    /* Force dropdown selectboxes and all child nodes to display click pointer cursor */
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] * {
        cursor: pointer !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ARTIFACT ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_churn_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model_artifact = load_model()

if model_artifact is None:
    st.error("⚠️ Operational Error: Model artifact reference unavailable.")
    st.stop()

model = model_artifact['model']
expected_features = model_artifact['features']

# --- HEADER SECTION ---
st.markdown('<p class="main-title">💠 ENTERPRISE CONSOLE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict customer retention risk in real-time with our optimized XGBoost Machine Learning Engine.</p>', unsafe_allow_html=True)

# --- SIDEBAR: CLEAN INPUT ARRAYS ---
st.sidebar.markdown("### Simulation Engine")
st.sidebar.caption("Adjust variables to simulate churn risk:")

with st.sidebar.expander("👤 Customer Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

with st.sidebar.expander("💳 Contract & Account Details", expanded=True):
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

with st.sidebar.expander("🌐 Subscribed Utilities", expanded=False):
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = tech_support = streaming_movies = "No internet service"

with st.sidebar.expander("📊 Financial Parameters", expanded=False):
    monthly_charges = st.number_input("Monthly Bill ($)", 15.0, 120.0, 65.0)
    total_charges = tenure * monthly_charges

# --- DATA PREPROCESSING ENGINE ---
def prepare_input_data():
    input_dict = {
        'gender': 1 if gender == "Female" else 0,
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
    df_input = pd.DataFrame([input_dict])
    multi_cols = ['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingMovies']
    df_encoded = pd.get_dummies(df_input, columns=multi_cols, drop_first=False)
    
    final_features = pd.DataFrame(0, index=[0], columns=expected_features)
    for col in df_encoded.columns:
        if col in final_features.columns:
            final_features[col] = df_encoded[col]
            
    return final_features.astype(float)

# --- RUN DIAGNOSTIC COMPUTATION ---
input_features = prepare_input_data()
churn_probability = model.predict_proba(input_features)[0][1] * 100

# --- MAIN DASHBOARD GRAPHICS ---
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("### Predictive Risk Analysis")
    
    # Custom Dark Mode Gauge Layout
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability,
        number={'suffix': "%", 'font': {'size': 44, 'color': '#F8FAFC', 'family': 'monospace'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Flight Risk Probability", 'font': {'size': 16, 'color': '#94A3B8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#64748B'}},
            'bar': {'color': "#F8FAFC", 'thickness': 0.15}, 
            'bgcolor': "#1E293B",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "rgba(46, 125, 50, 0.4)"},    
                {'range': [30, 70], 'color': "rgba(249, 168, 37, 0.3)"},   
                {'range': [70, 100], 'color': "rgba(198, 40, 40, 0.4)"}   
            ],
            'threshold': {
                'line': {'color': "#F8FAFC", 'width': 4},
                'thickness': 0.6,
                'value': churn_probability
            }
        }
    ))
    
    fig.update_layout(
        height=280, 
        margin=dict(l=30, r=30, t=40, b=10), 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("### Churn Risk Assessment")
    
    # Render customized executive cards with high-contrast font values
    if churn_probability >= 70:
        st.markdown(f"""
            <div class="protocol-card protocol-critical">
                <strong>Critical Risk Tier ({st.session_state.get('tier', '70%-100%')})</strong><br>
                Operational Status: IMMEDIATE INTERVENTION MANDATED. High probability of imminent contract cancellation.<br><br>
                <strong>VIP Account Escalation:</strong> Assign a dedicated senior customer success manager immediately. Issue proactive loyalty discounts or standard device credit adjustments.
            </div>
            """, unsafe_allow_html=True)
    elif churn_probability >= 30:
        st.markdown(f"""
            <div class="protocol-card protocol-warning">
                <strong>Elevated Risk Tier ({st.session_state.get('tier', '30%-70%')})</strong><br>
                Operational Status: CONDITIONAL MONITORING. Dynamic contract fragility flags triggered.<br><br>
                <strong>Mitigation Strategy:</strong> Queue account into the next automated product lifecycle campaign. Offer digital self-service upgrade paths.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="protocol-card protocol-stable">
                <strong>Stable Retention Tier ({st.session_state.get('tier', '0%-30%')})</strong><br>
                Operational Status: STANDARD LIFECYCLE. High core customer loyalty validation index.<br><br>
                <strong>Maintenance Track:</strong> Maintain standard operational support pathways. Eligible for multi-year renewal expansion upsells.
            </div>
            """, unsafe_allow_html=True)

st.divider()
# Replace the old st.dataframe block at the bottom with this:
st.markdown("### 🏷️ Active Configuration State")

badge_css = """
<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
    <span style='background-color: #1E293B; color: #F8FAFC; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #334155;'>⏱️ Tenure: {0}m</span>
    <span style='background-color: #1E293B; color: #F8FAFC; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #334155;'>📜 Contract: {1}</span>
    <span style='background-color: #1E293B; color: #F8FAFC; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #334155;'>🌐 Network: {2}</span>
    <span style='background-color: #1E293B; color: #F8FAFC; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #334155;'>💳 Rate: ${3}/mo</span>
    <span style='background-color: #1E293B; color: #F8FAFC; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #334155;'>📥 Billing: {4}</span>
</div>
"""
st.markdown(badge_css.format(tenure, contract, internet_service, monthly_charges, payment_method), unsafe_allow_html=True)

st.divider()
st.markdown("### 🔍 System Intelligence: How This Engine Evaluates Risk")
st.caption("A non-technical translation of the predictive patterns utilized by our XGBoost calculation framework:")

# Create clean interactive tabs for the business user
tab_contracts, tab_tenure, tab_infrastructure = st.tabs([
    "📜 The Power of Contracts", 
    "⏱️ Customer Lifespan (Tenure)", 
    "🌐 Infrastructure Impact"
])

with tab_contracts:
    st.markdown("""
    * **Two-Year Contracts act as the ultimate safety net.** Historically, locking a customer into a long-term agreement is the single strongest factor pulling their churn probability down toward 0%. It introduces a structural friction that heavily dampens sudden cancellations.
    * **Month-to-Month arrangements introduce high volatility.** Because there is zero financial or structural switching friction, the engine immediately elevates the baseline risk score for these accounts.
    """)

with tab_tenure:
    st.markdown("""
    * **The Onboarding 'Danger Zone':** The calculation engine heavily flags accounts within their first **1 to 3 months** of service. New customers haven't built brand loyalty yet, making them highly sensitive to initial service drops or billing surprises.
    * **The Retention Anchor:** As a customer's tenure grows past **12–24 months**, their historical baseline risk score decays significantly. Long-term customer accounts naturally exhibit high retention stability.
    """)

with tab_infrastructure:
    st.markdown("""
    * **The Fiber Optic Paradox:** Counter-intuitively, historical patterns show that premium **Fiber Optic** subscriptions exhibit a higher baseline churn risk than standard DSL lines. 
    * **Business Interpretation:** This signals a potential market anomaly—high-tier customers may be experiencing premium pricing exhaustion, or localized infrastructure reliability drops, making them highly volatile if proactive customer success touchpoints aren't maintained.
    """)

# Clean minimalist technical sub-footer
st.markdown("<br><hr><center style='color: #475569; font-size: 0.8rem;'>MODEL DEPLOYMENT: Active (v2.0.3) | Platform System Framework Architecture Stack</center>", unsafe_allow_html=True)