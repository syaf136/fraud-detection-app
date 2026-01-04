import time
import joblib
import pandas as pd
import streamlit as st
from fraud_preprocessor import FraudPreprocessor  # needed for joblib load

st.set_page_config(page_title="Fraud Detection (Real-time Dummy Stream)", page_icon="💳", layout="wide")
st.title("💳 Fraud Detection System — Dummy Real-time Streaming (Google Drive)")

@st.cache_resource
def load_artifacts():
    pre = joblib.load("artifacts/preprocessor.joblib")
    bundle = joblib.load("xgb_fraud_model.joblib")
    model = bundle["model"]
    feature_names = bundle.get("feature_names", None)
    return pre, model, feature_names

pre, model, feature_names = load_artifacts()

# -------- Load fraudTest.csv automatically from Google Drive --------
@st.cache_data(show_spinner=True)
def load_stream_data():
    url = "https://drive.google.com/uc?id=1uheCe1Z8Sb6zW0a6PB62upfsx81EdhJC"
    return pd.read_csv(url)

df = load_stream_data()

st.subheader("📄 Dummy Real-time Source: fraudTest.csv (Google Drive)")
st.write(f"Total rows available: {len(df)}")
st.dataframe(df.head(10), use_container_width=True)

# ---------------- Controls ----------------
st.sidebar.header("⚙️ Streaming Controls")
threshold = st.sidebar.slider("Fraud Threshold", 0.000001, 0.99, 0.50, 0.01)
speed = st.sidebar.slider("Seconds per transaction", 0.0, 2.0, 0.2, 0.1)
max_rows = st.sidebar.number_input("Rows to stream", min_value=1, max_value=min(5000, len(df)), value=min(200, len(df)))

start = st.button("▶️ Start Streaming")
reset = st.button("🔄 Reset")

if reset:
    st.rerun()

status = st.empty()
alert_box = st.empty()
table_box = st.empty()

if start:
    shown = []
    fraud_count = 0
    n = int(max_rows)

    for i in range(n):
        row_df = df.iloc[[i]]

        # preprocess + align
        X = pre.transform(row_df)
        if feature_names is not None:
            X = X.reindex(columns=feature_names, fill_value=0)

        # predict
        proba = float(model.predict_proba(X)[0, 1])
        pred = 1 if proba >= threshold else 0

        if pred == 1:
            fraud_count += 1
            alert_box.error(f"🚨 FRAUD ALERT | row={i} | probability={proba:.6f}")
        else:
            alert_box.success(f"✅ LEGIT | row={i} | probability={proba:.6f}")

        shown.append({
            "row": i,
            "probability": round(proba, 6),
            "prediction": "FRAUD" if pred == 1 else "LEGIT",
            "merchant": row_df.iloc[0].get("merchant", ""),
            "category": row_df.iloc[0].get("category", ""),
            "amt": row_df.iloc[0].get("amt", "")
        })

        status.info(f"Streamed: {i+1}/{n} | Fraud detected: {fraud_count}")
        table_box.dataframe(pd.DataFrame(shown).tail(15), use_container_width=True)

        time.sleep(speed)

    st.success("✅ Streaming finished.")
