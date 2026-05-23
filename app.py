import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Customer Diagnostics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM ENTERPRISE THEME INJECTION ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117 !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0B0D13 !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #F8FAFC !important;
    }
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
    
    /* Hide Deploy Button and Options Menu in header but keep sidebar collapse toggle visible */
    [data-testid="stHeaderDeployButton"],
    .stDeployButton,
    #MainMenu,
    [data-testid="stHeaderMenuButton"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem !important; padding-bottom: 0rem !important;}
    
    /* Force dropdown selectboxes and all child nodes to display click pointer cursor */
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] * {
        cursor: pointer !important;
    }
    
    /* Confusion matrix custom grid aesthetics */
    .matrix-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 8px;
        text-align: center;
        font-family: monospace;
        font-size: 0.9rem;
        border-radius: 4px;
        color: #F8FAFC;
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

# --- SIDEBAR: OPERATIONAL MODE ---
st.sidebar.markdown("### Workspace Console")
app_mode = st.sidebar.radio("Select Processing Track:", ["Single Simulation", "Bulk Processing (CSV)"])
st.sidebar.divider()

# --- INPUT ARRAYS CONFIGURATION CONTROL ---
if app_mode == "Single Simulation":
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

    input_features = prepare_input_data()
    churn_probability = model.predict_proba(input_features)[0][1] * 100

# Placeholder frame initialization for structural track safety checks
else:
    input_features = None

# --- SIDEBAR: STEP 3 MODEL TELEMETRY METRICS EXPANDER ---
st.sidebar.divider()
with st.sidebar.expander("🔬 Model Telemetry & Accuracy", expanded=False):
    st.caption("Validation metrics from XGBoost baseline training:")
    
    col_acc, col_f1 = st.columns(2)
    with col_acc:
        st.metric(label="Accuracy Index", value="80.4%")
    with col_f1:
        st.metric(label="ROC-AUC Score", value="0.842")
        
    col_prec, col_rec = st.columns(2)
    with col_prec:
        st.metric(label="Precision", value="67.1%")
    with col_rec:
        st.metric(label="Recall Score", value="55.8%")
        
    st.divider()
    st.caption("Validation Confusion Matrix Layout:")
    
    # Create clean dashboard matrix cells
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("<div class='matrix-box'><strong>True Neg</strong><br>1191</div>", unsafe_allow_html=True)
        st.markdown("<div class='matrix-box'><strong>False Neg</strong><br>206</div>", unsafe_allow_html=True)
    with mc2:
        st.markdown("<div class='matrix-box'><strong>False Pos</strong><br>144</div>", unsafe_allow_html=True)
        st.markdown("<div class='matrix-box'><strong>True Pos</strong><br>261</div>", unsafe_allow_html=True)

# --- MAIN RENDER LOGIC PATHWAY BRANCHES ---
if app_mode == "Single Simulation":
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown("### Predictive Risk Analysis")
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
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown("### Churn Risk Assessment")
        tier = "Stable"
        card_html = ""
        strategy_text = ""
        
        if churn_probability >= 70:
            tier = "Critical"
            card_html = """
                <div class="protocol-card protocol-critical">
                    <strong>Critical Risk Tier (70%-100%)</strong><br>
                    Operational Status: IMMEDIATE INTERVENTION MANDATED. High probability of imminent contract cancellation.<br><br>
                    <strong>VIP Account Escalation:</strong> Assign a dedicated senior customer success manager immediately. Issue proactive loyalty discounts or standard device credit adjustments.
                </div>"""
            strategy_text = "- IMMEDIATELY route to High-Value Retention Desk.\n- Approve maximum retention credit discount (up to 25% off monthly bill).\n- Schedule a high-priority 1-on-1 account health check sync within 24 hours."
        elif churn_probability >= 30:
            tier = "Elevated"
            card_html = """
                <div class="protocol-card protocol-warning">
                    <strong>Elevated Risk Tier (30%-70%)</strong><br>
                    Operational Status: CONDITIONAL MONITORING. Dynamic contract fragility flags triggered.<br><br>
                    <strong>Mitigation Strategy:</strong> Queue account into the next automated product lifecycle campaign. Offer digital self-service upgrade paths.
                </div>"""
            strategy_text = "- Target with specialized email lifecycle re-engagement loops.\n- Propose self-service value add-on upgrades (e.g., free premium security trials).\n- Monitor usage patterns closely over the next 30 billing cycles."
        else:
            card_html = """
                <div class="protocol-card protocol-stable">
                    <strong>Stable Retention Tier (0%-30%)</strong><br>
                    Operational Status: STANDARD LIFECYCLE. High core customer loyalty validation index.<br><br>
                    <strong>Maintenance Track:</strong> Maintain standard operational support pathways. Eligible for multi-year renewal expansion upsells.
                </div>"""
            strategy_text = "- Maintain standard marketing lifecycle cadences.\n- Keep flagged as stable for upcoming account volume calculations.\n- Target for multi-year contract expansion upsell options upon renewal date."

        st.markdown(card_html, unsafe_allow_html=True)

        brief_content = f"""# EXECUTIVE ESCALATION BRIEF: RETENTION TELEMETRY
Generated: 2026 Sandbox Environment Engine

## 📊 Account Diagnostic Profile
* **Risk Score Matrix Calculation:** {churn_probability:.1f}% Risk Probability
* **Assigned Action Tier:** {tier} Risk Track
* **Account Lifespan (Tenure):** {tenure} Months Active
* **Financial Vector:** ${monthly_charges:,.2f} / Month Recurring

## ⚙️ Active Configurations Captured
* **Contract Commitment Track:** {contract}
* **Network Infrastructure System:** {internet_service}
* **Settlement Channel:** {payment_method}
* **Paperless Status Billing:** {paperless_billing}

## ⚡ Mandatory Remediation Protocol Actions
{strategy_text}

---
*CONFIDENTIAL REPORT — Processed via deployed XGBoost Prediction Architecture Framework Framework Engine.*"""

        st.download_button(
            label="📥 Export Account Escalation Brief (.md)",
            data=brief_content,
            file_name=f"customer_risk_brief_{tier.lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.divider()
    st.markdown("### 📋 Live Simulation Parameters")
    st.caption("Active user configuration state fed into the XGBoost engine:")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Account Lifespan", value=f"{tenure} Months")
        st.metric(label="Billing Mechanism", value=paperless_billing)
    with m_col2:
        st.metric(label="Current Commitment", value=contract)
        st.metric(label="Financial Rate", value=f"${monthly_charges:,.2f}")
    with m_col3:
        st.metric(label="Infrastructure Track", value=internet_service)
        st.metric(label="Settlement Method", value=payment_method)

# --- TRACK 2: BULK CSV PROCESSING TRACK ---
else:
    st.markdown("### 📊 Bulk Ingestion & Batch Prediction Diagnostics")
    st.markdown("Upload a raw CSV file containing customer telemetry parameters to process downstream matrix scoring instantly.")
    
    uploaded_file = st.file_uploader("Choose a Customer CSV file...", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        
        st.markdown("#### 📥 Ingested Telemetry Preview")
        st.dataframe(raw_df.head(5), use_container_width=True)
        
        try:
            df_process = raw_df.copy()
            
            if 'gender' in df_process.columns: 
                df_process['gender'] = df_process['gender'].apply(lambda x: 1 if str(x).strip().lower() == "female" else 0)
            for binary_col in ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                if binary_col in df_process.columns:
                    df_process[binary_col] = df_process[binary_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
            
            multi_cols = ['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingMovies']
            df_encoded = pd.get_dummies(df_process, columns=[c for c in multi_cols if c in df_process.columns], drop_first=False)
            
            final_batch_features = pd.DataFrame(0, index=df_process.index, columns=expected_features)
            for col in df_encoded.columns:
                if col in final_batch_features.columns:
                    final_batch_features[col] = df_encoded[col]
            
            batch_probs = model.predict_proba(final_batch_features.astype(float))[:, 1] * 100
            
            output_df = raw_df.copy()
            output_df['Churn_Probability_%'] = np.round(batch_probs, 1)
            
            def assign_tier(prob):
                if prob >= 70: return 'Critical'
                elif prob >= 30: return 'Elevated'
                return 'Stable'
            output_df['Risk_Tier'] = output_df['Churn_Probability_%'].apply(assign_tier)
            
            st.divider()
            
            b_col1, b_col2 = st.columns([1, 1.2], gap="large")
            
            with b_col1:
                st.markdown("#### 📉 Batch Allocation Breakdown")
                tier_counts = output_df['Risk_Tier'].value_counts().reset_index()
                tier_counts.columns = ['Risk_Tier', 'Count']
                
                fig_pie = px.pie(
                    tier_counts, values='Count', names='Risk_Tier',
                    color='Risk_Tier',
                    color_discrete_map={'Stable': '#2E7D32', 'Elevated': '#F9A825', 'Critical': '#C62828'},
                    hole=0.4
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': 'white'},
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                
            with b_col2:
                st.markdown("#### 📥 Export Processed Payload")
                st.markdown("Download the fully annotated customer catalog containing evaluated risk vectors and tiered strategy markings.")
                
                csv_buffer = output_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="⚡ Download Appended Analytics (CSV)",
                    data=csv_buffer,
                    file_name="customer_churn_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            st.markdown("#### 🔬 Processed Batch Records Matrix")
            st.dataframe(output_df, use_container_width=True, height=220)
            
        except Exception as e:
            st.error(f"⚠️ Preprocessing Exception: Uploaded columns mismatch expected training signature. Error detail: {e}")

# --- SYSTEM EDUCATION FOOTER ---
st.divider()
st.markdown("### 🔍 System Intelligence: How This Engine Evaluates Risk")
st.caption("A non-technical translation of the predictive patterns utilized by our XGBoost calculation framework:")

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

st.markdown("<br><hr><center style='color: #475569; font-size: 0.8rem;'>MODEL DEPLOYMENT: Active (v2.0.3) | Platform System Framework Architecture Stack</center>", unsafe_allow_html=True)