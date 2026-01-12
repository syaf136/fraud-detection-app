import os
import time
import joblib
import pandas as pd
import streamlit as st
import gdown
from fraud_preprocessor import FraudPreprocessor  # needed for joblib load

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Fraud Detection (Real-time Stream)",
    page_icon="💳",
    layout="wide"
)
st.title("💳 Fraud Detection System — Simulation Real-time Streaming")

# ---------------- Data source (Google Drive) ----------------
FILE_ID = "1uheCe1Z8Sb6zW0a6PB62upfsx81EdhJC"
LOCAL_PATH = "data/fraudTest.csv"

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    pre = joblib.load("artifacts/preprocessor.joblib")
    bundle = joblib.load("xgb_fraud_model.joblib")
    model = bundle["model"]
    feature_names = bundle.get("feature_names", None)
    return pre, model, feature_names

# ---------------- Download + load data ----------------
@st.cache_data(show_spinner=True)
def download_and_load_data():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(LOCAL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, LOCAL_PATH, quiet=False)

    return pd.read_csv(LOCAL_PATH)

# ---------------- Sidebar controls ----------------
st.sidebar.header("⚙️ Streaming Controls")
st.sidebar.caption("Controls affect the simulation behavior, not model training settings.")

threshold = st.sidebar.slider(
    "Fraud Threshold (Probability)",
    min_value=0.000001,
    max_value=0.99,
    value=0.01,   # ✅ better default for fraud rarity
    step=0.01
)

speed = st.sidebar.slider(
    "Seconds per transaction",
    min_value=0.0,
    max_value=2.0,
    value=0.2,
    step=0.1
)

# ---------------- Load everything ----------------
st.info("Loading model + preprocessor...")
pre, model, feature_names = load_artifacts()
st.success("Model loaded ✅")

st.info("Loading dummy real-time data (fraudTest) ...")
df = download_and_load_data()
st.success(f"Data loaded ✅ Rows: {len(df)}")

max_rows = st.sidebar.number_input(
    "Rows to stream",
    min_value=1,
    max_value=min(5000, len(df)),
    value=min(200, len(df))
)

# ---------------- Layout (tabs) ----------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📄 Data Preview"])

with tab2:
    st.subheader("📄 Dummy Real-time Source Preview")
    st.dataframe(df.head(20), use_container_width=True)

with tab1:
    # Controls row
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        start = st.button("▶️ Start Streaming", use_container_width=True)
    with c2:
        reset = st.button("🔄 Reset", use_container_width=True)
    with c3:
        st.write(
            f"**Current Settings** → Threshold: `{threshold}` | Speed: `{speed}s/txn` | Max rows: `{max_rows}`"
        )

    if reset:
        st.cache_data.clear()
        st.rerun()

    # Live dashboard placeholders
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_total = metric_col1.empty()
    metric_fraud = metric_col2.empty()
    metric_legit = metric_col3.empty()

    status = st.empty()
    alert_box = st.empty()
    table_box = st.empty()

    if start:
        shown = []
        fraud_count = 0
        legit_count = 0
        n = int(max_rows)

        for i in range(n):
            row_df = df.iloc[[i]]

            # Preprocess
            X = pre.transform(row_df)
            if feature_names is not None:
                X = X.reindex(columns=feature_names, fill_value=0)

            # Predict
            proba = float(model.predict_proba(X)[0, 1])
            pred = 1 if proba >= threshold else 0

            if pred == 1:
                fraud_count += 1
                alert_box.error(f"🚨 FRAUD ALERT | row={i} | probability={proba:.6f}")
                label = "FRAUD"
            else:
                legit_count += 1
                alert_box.success(f"✅ LEGIT | row={i} | probability={proba:.6f}")
                label = "LEGIT"

            # Add to rolling log
            shown.append({
                "row": i,
                "probability": round(proba, 6),
                "prediction": label,
                "merchant": row_df.iloc[0].get("merchant", ""),
                "category": row_df.iloc[0].get("category", ""),
                "amt": row_df.iloc[0].get("amt", "")
            })

            # Update metrics
            metric_total.metric("Streamed", i + 1)
            metric_fraud.metric("Fraud Detected", fraud_count)
            metric_legit.metric("Legit", legit_count)

            status.info(f"Streaming: {i+1}/{n}")

            # Show last 20 rows like a live monitor
            table_box.dataframe(pd.DataFrame(shown).tail(20), use_container_width=True)

            time.sleep(speed)

        st.success("✅ Streaming finished.")
