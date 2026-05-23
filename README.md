# 💠 Proactive Customer Retention Portal & Risk Diagnostics Console

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML%20Engine-XGBoost%202.0-F9A825?style=flat-square)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Visuals-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

An end-to-end predictive diagnostics console built using an optimized **XGBoost Machine Learning pipeline** to simulate, detect, and preempt customer churn. Designed as a high-fidelity business intelligence panel, it translates complex ML probabilities into immediate, risk-tier-matched retention protocols for customer success teams.

---

## 🚀 Core Capabilities

The architecture operates on a **dual-track processing framework** tailored to both immediate diagnostic simulations and large-scale, automated account audits:

### 1. Single-Customer Simulation Sandbox
- **Speedometer-Style Visuals:** Displays calculated risk probabilities in a high-contrast Plotly gauge chart complete with custom indicator zones (Stable green, Warning yellow, Critical red).
- **Interactive Simulation Controls:** Allows account managers to adjust demographic, service subscription, and billing configuration sliders in real-time to analyze how account changes impact risk indices.
- **Dynamic Executive Briefing Generator:** Automatically compiles customer parameters, calculated metrics, and tier-specific strategic retention checklists into a downloadable, clean **Markdown (.md) briefing file** for immediate CRM logging.

<p align="center">
  <img src="assests/dashboard_mockup.png" width="850" alt="UI Showcase - Single Customer Simulation">
</p>

### 2. Bulk Ingestion & Batch Prediction Engine
- **Automated Preprocessing:** Accepts drag-and-drop raw CSV customer database uploads, mapping categorical indicators and executing one-hot feature encoding under the hood.
- **Batch Diagnostic Charts:** Renders interactive, multi-dimensional Plotly donut charts to visualize risk tier allocations across the entire uploaded segment.
- **Strategic Payload Exports:** Performs vector calculations and exports a fully annotated CSV data sheet mapping baseline attributes directly against predictions and mitigation instructions.

<p align="center">
  <img src="assests/batch_processing.png" width="850" alt="UI Showcase - Bulk Batch Processing">
</p>

---

## 🛠️ The Core Stack

The system utilizes highly optimized libraries to deliver fast predictions and rich interactive components:

| Tech Component | Standard Reference | Operational Purpose |
| :--- | :--- | :--- |
| **Python** | `v3.9+` | Underlying programmatic programming execution layer. |
| **Streamlit** | `v1.41.1` | Web framework hosting dark-themed SaaS interface console. |
| **XGBoost** | `v2.0.3` | Tuned machine learning classifier engine using advanced gradient boosting tree algorithms. |
| **Plotly** | `v6.7.0` | Renders radial indicator gauge speedometers and batch pie/donut distribution charts. |
| **Pandas** | `v2.2.3` | Matrix manipulation, telemetry table cleaning, and feature schema processing. |
| **NumPy** | `v1.26.4` | Vector calculations, batch prob mapping, and float rounding checks. |
| **Joblib** | `v1.4.2` | Serialized model loading and XGBoost weight artifact deserialization. |

---

## 🔬 Model Validation Telemetry

The baseline machine learning engine was trained on a dataset of customer profiles, leveraging minority-class weighting to enhance classification recall.

### Core Metrics Attained
- **Accuracy Index:** `80.4%` — General predictive stability score.
- **ROC-AUC Score:** `0.842` — Superior class-separation probability index.
- **Precision Score:** `67.1%` — Reliability rating of flagged risk positives.
- **Recall Index:** `55.8%` — Captured percentage of the total churning customer population.

### Validation Confusion Matrix

| | Predicted Stable (0) | Predicted Churn (1) |
| :--- | :---: | :---: |
| **Actual Stable (0)** | **1191** (True Negative) | **144** (False Positive) |
| **Actual Churn (1)** | **206** (False Negative) | **261** (True Positive) |

---

## 🧠 System Intelligence & Data Intuition

The predictive triggers utilized by the XGBoost engine are rooted in underlying business behaviors, explained through global SHAP feature attribution parameters:

1. **The Power of Contracts (Contract Friction):** Two-Year agreements provide the ultimate safety net, creating structural friction that dampens churn. In contrast, Month-to-Month arrangements introduce zero financial switching costs, driving high account volatility.
2. **The Onboarding Danger Zone (Lifespan vs. Price):** Churn indicators peak heavily within the first **1 to 3 months** of customer onboarding, particularly on high-tier plans billing **$70+/month**. Proactive customer success check-ins must be targeted at these new, high-value cohorts.
3. **The Fiber Optic Premium Paradox:** Counter-intuitively, high-speed **Fiber Optic** packages exhibit higher baseline churn risks than DSL. This suggests premium pricing fatigue or localized network quality volatility, which necessitates targeted value reinforcement.

---

## 💻 Local Deployment Blueprint

To deploy the Enterprise Churn Diagnostics Portal locally on your machine, follow these terminal instructions:

### 1. Prepare Virtual Workspace
Create a clean local Python virtual environment to prevent package collisions:
```powershell
# Create environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Install Project Dependencies
Verify that Plotly, Streamlit, and your computing libraries are installed:
```powershell
pip install -r requirements.txt
```

### 3. Launch the SaaS Console
Fire up the local Streamlit development server:
```powershell
python -m streamlit run app.py
```
Upon startup, the console will be accessible in your web browser at `http://localhost:8501`.