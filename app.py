import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gdown

# Needed for joblib load (custom class inside joblib)
from fraud_preprocessor import FraudPreprocessor

# =========================
# Page config + theme
# =========================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# --- Dark dashboard styling (close to your example) ---
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] { background: #0b0f14; }
      [data-testid="stSidebar"] { background: #0a0d12; border-right: 1px solid #1f2a37; }
      [data-testid="stHeader"] { background: rgba(0,0,0,0); }
      .block-container { padding-top: 1rem; }

      .status-ok { background:#0f2b1f; border:1px solid #1d5b3f; color:#a7f3d0; padding:12px 16px; border-radius:10px; }
      .status-info { background:#0b1f33; border:1px solid #1e3a5f; color:#93c5fd; padding:12px 16px; border-radius:10px; }
      .card { background:#0f172a; border:1px solid #233146; border-radius:14px; padding:16px; }
      .title { font-size: 34px; font-weight: 800; color: #e5e7eb; }
      .subtle { color: #9ca3af; }
      .pill { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #2b3444; color:#cbd5e1; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* GLOBAL TEXT COLORS */
    html, body, [class*="css"]  {
        color: #f9fafb !important;   /* near-white */
    }

    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f14;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0d12;
        border-right: 1px solid #1f2937;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Normal text */
    p, span, label, div {
        color: #e5e7eb !important;
    }

    /* Streamlit metrics */
    [data-testid="stMetricLabel"] {
        color: #cbd5f5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px;
        font-weight: 700;
    }

    /* Cards / containers */
    .card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px;
        color: #f8fafc !important;
    }

    /* Success / Info banners */
    .status-ok {
        background: #064e3b;
        border: 1px solid #10b981;
        color: #ecfdf5 !important;
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 600;
    }

    .status-info {
        background: #0c4a6e;
        border: 1px solid #38bdf8;
        color: #e0f2fe !important;
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 600;
    }

    /* Tables */
    .stDataFrame {
        color: #ffffff !important;
    }

    /* Buttons */
    button {
        color: #ffffff !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True

    /* === ANALYZE BUTTON STYLE === */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.75rem 1.2rem !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.6);
}

/* Hover effect */
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af, #1e3a8a) !important;
    box-shadow: 0 0 18px rgba(37, 99, 235, 0.9);
    transform: translateY(-1px);
}

/* Active / clicked */
button[kind="primary"]:active {
    background: #1e3a8a !important;
    transform: translateY(0px);
}

/* Disabled Analyze button */
button:disabled {
    background: #334155 !important;
    color: #cbd5e1 !important;
    box-shadow: none !important;
}

)


st.markdown("<div class='title'>🔎 Real-Time Fraud Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>Test the fraud detection system with real or simulated transactions. "
    "This prototype streams transactions and raises alerts using an ML probability threshold.</div>",
    unsafe_allow_html=True
)

# =========================
# Google Drive data source
# =========================
FILE_ID = "1uheCe1Z8Sb6zW0a6PB62upfsx81EdhJC"
LOCAL_PATH = "data/fraudTest.csv"

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    pre = joblib.load("artifacts/preprocessor.joblib")
    bundle = joblib.load("artifacts/xgb_fraud_model.joblib") if os.path.exists("artifacts/xgb_fraud_model.joblib") else joblib.load("xgb_fraud_model.joblib")
    model = bundle["model"]
    feature_names = bundle.get("feature_names", None)
    return pre, model, feature_names

@st.cache_data(show_spinner=True)
def download_and_load_default_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOCAL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, LOCAL_PATH, quiet=False)
    return pd.read_csv(LOCAL_PATH)

def prepare_features(pre, model_feature_names, raw_df: pd.DataFrame):
    X = pre.transform(raw_df)
    if model_feature_names is not None:
        # ensure same column order as training
        X = X.reindex(columns=model_feature_names, fill_value=0)
    return X

# =========================
# Sidebar (Control Panel)
# =========================
st.sidebar.markdown("## ⚙️ Control Panel")
mode = st.sidebar.radio(
    "Select Mode",
    ["📊 Dashboard Overview", "🔎 Real-time Detection", "📈 Analytics", "⚡ Performance Metrics"]
)

st.sidebar.divider()

st.sidebar.markdown("### 🔐 Security Status")
st.sidebar.success("Encryption: ENABLED")

st.sidebar.markdown("### 🎛️ Detection Settings")
threshold = st.sidebar.slider(
    "Fraud Threshold (Probability)",
    min_value=0.000001,
    max_value=0.99,
    value=0.01,
    step=0.01
)
speed = st.sidebar.slider(
    "Seconds per transaction",
    min_value=0.0,
    max_value=2.0,
    value=0.2,
    step=0.1
)

# =========================
# Load model + default data
# =========================
top_msg1 = st.empty()
top_msg2 = st.empty()
top_msg3 = st.empty()

top_msg1.markdown("<div class='status-info'>Loading model + preprocessor...</div>", unsafe_allow_html=True)
pre, model, feature_names = load_artifacts()
top_msg1.markdown("<div class='status-ok'>Model loaded successfully ✅</div>", unsafe_allow_html=True)

top_msg2.markdown("<div class='status-info'>Loading dummy real-time data (fraudTest)...</div>", unsafe_allow_html=True)
default_df = download_and_load_default_data()
top_msg2.markdown(f"<div class='status-ok'>Data loaded ✅ Rows: {len(default_df)}</div>", unsafe_allow_html=True)

# =========================
# Helpers: summary from probabilities
# =========================
def predict_proba_for_df(df_in: pd.DataFrame):
    X = prepare_features(pre, feature_names, df_in)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

def build_result_df(raw_df: pd.DataFrame, proba: np.ndarray, pred: np.ndarray):
    out = raw_df.copy()
    out["fraud_probability"] = proba
    out["prediction"] = np.where(pred == 1, "FRAUD", "LEGIT")
    return out

# =========================
# Main Pages
# =========================

# ---------- Dashboard Overview ----------
if mode == "📊 Dashboard Overview":
    st.markdown("## 📊 Dashboard Overview")
    st.markdown("<span class='pill'>Default source: fraudTest</span>", unsafe_allow_html=True)
    st.write("")

    # Light analytics based on default_df (fast preview)
    sample_n = min(2000, len(default_df))
    sample_df = default_df.head(sample_n)
    proba_s, pred_s = predict_proba_for_df(sample_df)

    total = int(len(sample_df))
    fraud_count = int(pred_s.sum())
    legit_count = total - fraud_count
    fraud_rate = (fraud_count / total) * 100 if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions (sample)", f"{total:,}")
    c2.metric("Detected Fraud (sample)", f"{fraud_count:,}")
    c3.metric("Fraud Rate (sample)", f"{fraud_rate:.2f}%")
    c4.metric("Threshold", f"{threshold}")

    st.markdown("### 🔎 Recent Predictions (sample)")
    preview = build_result_df(sample_df.head(50), proba_s[:50], pred_s[:50])
    st.dataframe(preview, use_container_width=True)

    st.info("Note: Overview uses a sample for speed. Use **Real-time Detection** to stream transactions.")

# ---------- Real-time Detection ----------
elif mode == "🔎 Real-time Detection":
    st.markdown("## 🔎 Real-time Detection")
    st.markdown("<span class='pill'>3 input modes</span>", unsafe_allow_html=True)
    st.write("")

    input_method = st.radio(
        "Input Method",
        ["🎲 Random from Default Dataset", "📌 By Rows (Stream)", "📤 Upload Another Dataset"],
        horizontal=True
    )

    st.divider()

    # ========== 1) Random ==========
    if input_method == "🎲 Random from Default Dataset":
        left, right = st.columns([2, 1])

        with left:
            st.markdown("### 🎲 Random Transaction")
            st.markdown("<div class='subtle'>Pick a random row from fraudTest and run prediction.</div>", unsafe_allow_html=True)

            if st.button("Analyze Random Transaction", use_container_width=True):
                idx = np.random.randint(0, len(default_df))
                row_df = default_df.iloc[[idx]]

                X = prepare_features(pre, feature_names, row_df)
                proba = float(model.predict_proba(X)[0, 1])
                pred = 1 if proba >= threshold else 0

                st.caption(f"Selected row index: {idx}")
                st.dataframe(row_df, use_container_width=True)

                if pred == 1:
                    st.error(f"🚨 FRAUD DETECTED | Probability = {proba:.6f}")
                else:
                    st.success(f"✅ LEGIT | Probability = {proba:.6f}")

        with right:
            st.markdown("### ℹ️ Tips")
            st.write("- Use a lower threshold (e.g., 0.01) to see more alerts.")
            st.write("- Random mode is good for quick testing.")

    # ========== 2) By rows (stream like earlier system) ==========
    elif input_method == "📌 By Rows (Stream)":
        st.markdown("### 📌 Stream by Row Range")
        st.markdown("<div class='subtle'>Simulate real-time incoming transactions from a chosen start row.</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            start_row = st.number_input("Start row (0-based)", min_value=0, max_value=len(default_df) - 1, value=0)
        with c2:
            max_rows = st.number_input("Rows to stream", min_value=1, max_value=min(5000, len(default_df)), value=min(200, len(default_df)))
        with c3:
            st.write("")
            stream_btn = st.button("▶️ Start Streaming", use_container_width=True)

        reset_btn = st.button("🔄 Reset View", use_container_width=False)
        if reset_btn:
            st.rerun()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_total = metric_col1.empty()
        metric_fraud = metric_col2.empty()
        metric_legit = metric_col3.empty()

        status = st.empty()
        alert_box = st.empty()
        table_box = st.empty()

        if stream_btn:
            shown = []
            fraud_count = 0
            legit_count = 0

            n = int(max_rows)
            start_i = int(start_row)

            for i in range(start_i, start_i + n):
                if i >= len(default_df):
                    break

                row_df = default_df.iloc[[i]]
                X = prepare_features(pre, feature_names, row_df)

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

                shown.append({
                    "row": i,
                    "probability": round(proba, 6),
                    "prediction": label,
                    "merchant": row_df.iloc[0].get("merchant", ""),
                    "category": row_df.iloc[0].get("category", ""),
                    "amt": row_df.iloc[0].get("amt", "")
                })

                metric_total.metric("Streamed", len(shown))
                metric_fraud.metric("Fraud Detected", fraud_count)
                metric_legit.metric("Legit", legit_count)

                status.info(f"Streaming: {len(shown)}/{n}")
                table_box.dataframe(pd.DataFrame(shown).tail(20), use_container_width=True)

                time.sleep(speed)

            st.success("✅ Streaming finished.")

    # ========== 3) Upload dataset ==========
    else:
        st.markdown("### 📤 Upload Another Dataset (CSV)")
        st.markdown("<div class='subtle'>Upload a CSV file and run batch fraud prediction.</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded dataset loaded ✅ Rows: {len(new_df)}")
                st.dataframe(new_df.head(20), use_container_width=True)

                if st.button("Run Prediction on Uploaded Dataset", use_container_width=True):
                    proba, pred = predict_proba_for_df(new_df)
                    result = build_result_df(new_df, proba, pred)

                    fraud_total = int((pred == 1).sum())
                    st.warning(f"🚨 Total Fraud Detected: {fraud_total:,} / {len(result):,}")

                    st.dataframe(result.head(100), use_container_width=True)

                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Results CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Error reading file or predicting: {e}")

# ---------- Analytics ----------
elif mode == "📈 Analytics":
    st.markdown("## 📈 Analytics")
    st.markdown("<span class='pill'>Based on default dataset sample</span>", unsafe_allow_html=True)
    st.write("")

    sample_n = min(5000, len(default_df))
    sample_df = default_df.head(sample_n)
    proba, pred = predict_proba_for_df(sample_df)

    st.markdown("### Fraud Probability Distribution (sample)")
    chart_df = pd.DataFrame({"fraud_probability": proba})
    st.line_chart(chart_df["fraud_probability"].reset_index(drop=True))

    st.markdown("### Fraud vs Legit Counts (sample)")
    counts = pd.DataFrame({
        "label": ["LEGIT", "FRAUD"],
        "count": [int((pred == 0).sum()), int((pred == 1).sum())]
    }).set_index("label")
    st.bar_chart(counts)

# ---------- Performance Metrics ----------
else:
    st.markdown("## ⚡ Performance Metrics")
    st.markdown("<div class='subtle'>These are deployment-time monitoring metrics (not full training evaluation).</div>", unsafe_allow_html=True)
    st.write("")

    # Use a small sample for quick reporting
    sample_n = min(5000, len(default_df))
    sample_df = default_df.head(sample_n)
    proba, pred = predict_proba_for_df(sample_df)

    st.markdown("### Threshold Summary (sample)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold", f"{threshold}")
    c2.metric("Sample size", f"{sample_n:,}")
    c3.metric("Predicted fraud", f"{int(pred.sum()):,}")

    st.info(
        "For true model performance metrics (Accuracy/Recall/ROC-AUC), report the values from your training/testing evaluation scripts."
    )

# ---------- Data Preview (always at bottom) ----------
with st.expander("📄 View default dataset preview (fraudTest)"):
    st.dataframe(default_df.head(30), use_container_width=True)



