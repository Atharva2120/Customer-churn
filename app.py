import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction")
st.write("Upload a CSV file to predict customer churn using the trained model.")

model_path = st.sidebar.text_input("Model path", "outputs/model.pkl")
metrics_path = st.sidebar.text_input("Metrics path", "outputs/metrics.json")
fi_path = st.sidebar.text_input("Feature importance", "outputs/feature_importance.csv")
eda_dir = st.sidebar.text_input("EDA directory", "outputs/eda")

if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    st.sidebar.subheader("Model Metrics")
    st.sidebar.json(metrics)

model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.warning("Model not found. Train a model first using src/train.py.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

target_col = st.text_input("Target column (optional; will be dropped if present)", "Churn")

if uploaded and model is not None:
    df = pd.read_csv(uploaded)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None

    results = df.copy()
    results["churn_prediction"] = preds
    if proba is not None:
        results["churn_probability"] = proba

    st.subheader("Predictions")
    st.dataframe(results.head(20))

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

if os.path.exists(fi_path):
    st.subheader("Feature Importance")
    fi = pd.read_csv(fi_path).head(20)
    st.bar_chart(fi.set_index("feature")["importance"])

if os.path.isdir(eda_dir):
    st.subheader("EDA Plots")
    for filename in ["target_distribution.png", "numeric_histograms.png", "correlation_heatmap.png"]:
        path = os.path.join(eda_dir, filename)
        if os.path.exists(path):
            st.image(path, caption=filename)
