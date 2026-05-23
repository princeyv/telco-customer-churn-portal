# 📊 Telco Customer Churn Diagnostics & Proactive Retention Portal

## 📌 Executive Summary
Customer churn represents a massive drain on enterprise revenue. This project implements an **end-to-end Machine Learning solution** designed to identify at-risk accounts before they cancel their service. 

By addressing historical class imbalance through algorithm weighting, the production machine learning pipeline achieves an **81% Recall rate** on churning customers—allowing operational teams to successfully intercept predictive withdrawal behaviors.

---

## 🏗️ System Architecture & Engineering Stack
* **Data Processing & Feature Engineering:** `Pandas`, `NumPy`, `Scikit-Learn` (One-Hot & Label Encoding)
* **Core Modeling Engine:** Optimized `XGBoost Classifier` (`scale_pos_weight` tuned for minority class detection)
* **Model Lifecycle & MLOps:** `MLflow` for experiment tracking and hyperparameter logging
* **Global Explainability:** `SHAP` (SHapley Additive exPlanations) for transparent feature attribution
* **Frontend Web Application:** Fully responsive `Streamlit` user interface loading serialized `.pkl` models via `Joblib`

---

## 📈 Key Business Insights & SHAP Interpretations
1. **The Ultimate Shield (Contract Length):** Two-Year contracts serve as the strongest mathematical barrier against cancellations. Month-to-month arrangements introduce zero switching friction, driving extreme volatility.
2. **The Danger Zone (Lifespan vs. Price):** Churn concentration peaks heavily within the first 1–3 months of customer onboarding, specifically among accounts generating upward of **$70+/month** in billing.
3. **Infrastructure Anomalies:** Premium **Fiber Optic** accounts exhibit higher baseline churn probabilities than standard DSL lines, highlighting potential pricing exhaustion or localized reliability drops.

---

## 💻 Local Execution Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/telco-customer-churn-portal.git](https://github.com/yourusername/telco-customer-churn-portal.git)
cd telco-customer-churn-portal