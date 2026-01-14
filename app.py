import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st 
import gdown
import plotly.express as px

# ---- Plotly gauge (safe import) ----
try:
    import plotly.express as px  # not used directly, but keep if installed
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Needed for joblib load (custom class inside joblib)
from fraud_preprocessor import FraudPreprocessor

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# =========================
# CSS (Theme + Buttons + Text)
# =========================
st.markdown(
    """
    <style>
    /* ---------- App background + sidebar ---------- */
    [data-testid="stAppViewContainer"] { background: #0b0f14; }
    [data-testid="stSidebar"] { background: #0a0d12; border-right: 1px solid #1f2937; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container { padding-top: 1rem; }

    /* ---------- Global text colors ---------- */
    html, body, [class*="css"]  { color: #f9fafb !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 800; }
    p, span, label, div { color: #e5e7eb !important; }

    /* =========================
       FILE UPLOADER (Dark Theme)
       ========================= */
    [data-testid="stFileUploader"] {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #111827 !important;
        border: 2px dashed #374151 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        fill: #e5e7eb !important;
        color: #e5e7eb !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #60a5fa !important;
        background: #0b1220 !important;
    }

    /* ---------- Metric colors ---------- */
    [data-testid="stMetricLabel"] { color: #cbd5f5 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px; font-weight: 800; }

    /* ---------- Status banners ---------- */
    .status-ok {
        background: #064e3b;
        border: 1px solid #10b981;
        color: #ecfdf5 !important;
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 700;
    }
    .status-info {
        background: #0c4a6e;
        border: 1px solid #38bdf8;
        color: #e0f2fe !important;
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 700;
    }

    /* ---------- Title helpers ---------- */
    .title { font-size: 34px; font-weight: 900; color: #ffffff; }
    .subtle { color: #9ca3af !important; }
    .pill {
        display:inline-block; padding:4px 10px; border-radius:999px;
        border:1px solid #2b3444; color:#cbd5e1 !important; font-size:12px;
    }

    /* =========================
       BUTTON STYLES (PRIMARY)
       ========================= */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 1.2rem !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.6);
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e40af, #1e3a8a) !important;
        box-shadow: 0 0 18px rgba(37, 99, 235, 0.9);
        transform: translateY(-1px);
    }
    button[kind="primary"]:active {
        background: #1e3a8a !important;
        transform: translateY(0px);
    }
    button[kind="primary"]:disabled {
        background: #1f2937 !important;
        color: #9ca3af !important;
        border: 1px solid #374151 !important;
        box-shadow: none !important;
        opacity: 1 !important;
        cursor: not-allowed !important;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    model_path = "artifacts/xgb_fraud_model.joblib"
    bundle = joblib.load(model_path) if os.path.exists(model_path) else joblib.load("xgb_fraud_model.joblib")
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

# =========================
# Helpers
# =========================
LABEL_CANDIDATES = ["is_fraud", "Class", "label", "target"]

def find_label_col(df: pd.DataFrame):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def drop_label_cols(df: pd.DataFrame):
    """Drop label columns so model never receives them."""
    label_col = find_label_col(df)
    if label_col is None:
        return df.copy(), None
    return df.drop(columns=[label_col]).copy(), label_col

def prepare_features(pre, model_feature_names, raw_df: pd.DataFrame):
    """
    IMPORTANT:
    - Ensure label column (is_fraud etc) is NOT used for prediction
    """
    feature_df, _ = drop_label_cols(raw_df)
    X = pre.transform(feature_df)
    if model_feature_names is not None:
        X = X.reindex(columns=model_feature_names, fill_value=0)
    return X

def predict_one_row(row_df: pd.DataFrame, threshold: float):
    t0 = time.perf_counter()
    X = prepare_features(pre, feature_names, row_df)
    proba = float(model.predict_proba(X)[0, 1])
    pred = 1 if proba >= threshold else 0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return proba, pred, elapsed_ms

def predict_proba_for_df(df_in: pd.DataFrame, threshold: float):
    X = prepare_features(pre, feature_names, df_in)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

def build_result_df(raw_df: pd.DataFrame, proba: np.ndarray, pred: np.ndarray):
    out = raw_df.copy()
    out["fraud_probability"] = proba
    out["prediction"] = np.where(pred == 1, "FRAUD", "LEGIT")
    return out

def confidence_level(proba: float, threshold: float):
    # Simple, defensible UI logic (not model logic)
    if proba >= threshold:
        return "High" if proba >= 0.80 else "Medium"
    else:
        return "High" if proba <= 0.05 else "Medium"

def plot_risk_gauge(risk_pct: float):
    """
    Gauge = fraud_probability * 100
    Uses Plotly if available; otherwise fallback to a simple metric.
    """
    if not PLOTLY_OK:
        st.metric("Risk Level", f"{risk_pct:.1f}%")
        return

    fig = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number",
            "value": risk_pct,
            "number": {"suffix": "%", "font": {"size": 46, "color": "white"}},
            "gauge": {
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "#60a5fa"},
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 70], "color": "#854d0e"},
                    {"range": [70, 100], "color": "#7f1d1d"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            "title": {"text": "Risk Level", "font": {"size": 20, "color": "white"}},
        }],
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "height": 260,
        }
    }
    st.plotly_chart(fig, use_container_width=True)

def show_analysis_panel(row_df: pd.DataFrame, proba: float, pred: int, elapsed_ms: float, threshold: float):
    """
    Shows exactly the requested Analysis panel:
    - Fraud Probability (Model)
    - Decision Threshold
    - Prediction
    - Detection Time
    - Confidence Level
    - Gauge = proba*100
    """
    # Transaction fields (best-effort)
    t = row_df.iloc[0].to_dict()
    merch = t.get("merchant", "—")
    cat = t.get("category", "—")
    amt = t.get("amt", "—")
    loc = t.get("city", t.get("state", "—"))

    risk_pct = proba * 100.0
    threshold_pct = threshold * 100.0
    conf = confidence_level(proba, threshold)
    pred_label = "FRAUD" if pred == 1 else "LEGIT"

    st.markdown("### 🔍 Analysis")

    a, b, c = st.columns([1.4, 1.3, 1.3])

    with a:
        st.markdown("#### Transaction Details")
        st.write(f"**Merchant:** {merch}")
        st.write(f"**Category:** {cat}")
        st.write(f"**Amount:** ${amt}" if isinstance(amt, (int, float, np.number)) else f"**Amount:** {amt}")
        st.write(f"**Location:** {loc}")

    with b:
        st.markdown("#### Detection Metrics")
        st.write(f"**Fraud Probability (Model):** {risk_pct:.1f}%")
        st.write(f"**Decision Threshold:** {threshold_pct:.1f}%")
        st.write(f"**Prediction:** {pred_label}")
        st.write(f"**Detection Time:** {elapsed_ms:.2f} ms")
        st.write(f"**Confidence Level:** {conf}")

    with c:
        plot_risk_gauge(risk_pct)

# =========================
# Sidebar
# =========================
st.sidebar.markdown("## ⚙️ Control Panel")
mode = st.sidebar.radio("Select Mode", ["📊 Dashboard Overview", "🔎 Real-time Detection"])

st.sidebar.markdown("### 🎛️ Detection Settings")
threshold = st.sidebar.slider(
    "Fraud Threshold (Probability)",
    min_value=0.000001,
    max_value=0.99,
    value=0.01,
    step=0.01
)
speed = st.sidebar.slider("Seconds per transaction", 0.0, 2.0, 0.2, 0.1)

# =========================
# Load model + default data
# =========================
top_msg1 = st.empty()
top_msg2 = st.empty()

top_msg1.markdown("<div class='status-info'>Loading model + preprocessor...</div>", unsafe_allow_html=True)
pre, model, feature_names = load_artifacts()
top_msg1.markdown("<div class='status-ok'>Model loaded successfully ✅</div>", unsafe_allow_html=True)

top_msg2.markdown("<div class='status-info'>Loading dummy real-time data (fraudTest)...</div>", unsafe_allow_html=True)
default_df = download_and_load_default_data()
top_msg2.markdown(f"<div class='status-ok'>Data loaded ✅ Rows: {len(default_df)}</div>", unsafe_allow_html=True)

# =========================
# Pages
# =========================

if mode == "📊 Dashboard Overview":
    import matplotlib.pyplot as plt

    st.markdown("## 📊 Dashboard Overview")
    st.markdown("<span class='pill'>Default source: fraudTest</span>", unsafe_allow_html=True)
    st.write("")

    # Use a sample for speed (change size if you want)
    sample_n = min(50000, len(default_df))
    sample_df = default_df.head(sample_n)

    # Predict on sample (label is dropped inside prepare_features)
    proba_s, pred_s = predict_proba_for_df(sample_df, threshold)

    total = int(len(sample_df))
    fraud_count = int(pred_s.sum())
    fraud_rate = (fraud_count / total) * 100 if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions (sample)", f"{total:,}")
    c2.metric("Detected Fraud (sample)", f"{fraud_count:,}")
    c3.metric("Fraud Rate (sample)", f"{fraud_rate:.2f}%")
    c4.metric("Threshold", f"{threshold:.2f}")

    st.divider()

    colL, colR = st.columns(2)

    # ==========================================================
    # Graph 1: Transaction Amount Distribution (BINS + LABELS)
    # ==========================================================
    with colL:
        st.subheader("Transaction Amount Distribution")
        st.caption("Amount distribution by predicted fraud status")

        if "amt" not in sample_df.columns:
            st.warning("Column `amt` not found in dataset.")
        else:
            tmp = sample_df.copy()
            tmp["pred_label"] = np.where(pred_s == 1, "FRAUD", "LEGIT")

            # bins in USD (since fraudTest amt is USD)
            bins = [0, 10, 50, 100, 200, 500, 1000, np.inf]
            labels = ["0–10", "10–50", "50–100", "100–200", "200–500", "500–1000", ">1000"]

            tmp["amt_bin"] = pd.cut(tmp["amt"].astype(float), bins=bins, labels=labels, include_lowest=True)

            # count per bin per label
            pivot = (
                tmp.groupby(["amt_bin", "pred_label"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(labels)
            )

            # Plot
            fig, ax = plt.subplots()
            x = np.arange(len(labels))
            width = 0.40

            fraud_vals = pivot["FRAUD"].values if "FRAUD" in pivot.columns else np.zeros(len(labels))
            legit_vals = pivot["LEGIT"].values if "LEGIT" in pivot.columns else np.zeros(len(labels))

            ax.bar(x - width/2, fraud_vals, width, label="FRAUD")
            ax.bar(x + width/2, legit_vals, width, label="LEGIT")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_xlabel("Transaction Amount Range (USD)")
            ax.set_ylabel("Number of Transactions")
            ax.legend()

            st.pyplot(fig)

    # ==========================================================
    # Graph 2: Fraud Rate by Category (Top 15)
    # ==========================================================
    with colR:
        st.subheader("Fraud by Merchant Category")
        st.caption("Predicted fraud rate (%) by category (top 15)")

        if "category" not in sample_df.columns:
            st.warning("Column `category` not found in dataset.")
        else:
            tmp = sample_df.copy()
            tmp["pred"] = pred_s.astype(int)

            rate = (
                tmp.groupby("category")["pred"]
                   .mean()
                   .mul(100)
                   .sort_values(ascending=False)
                   .head(15)
            )

            fig, ax = plt.subplots()
            ax.bar(rate.index.astype(str), rate.values)
            ax.set_xlabel("Merchant Category")
            ax.set_ylabel("Fraud Rate (%)")
            ax.set_title("Top 15 Categories by Predicted Fraud Rate")
            plt.xticks(rotation=30, ha="right")

            st.pyplot(fig)

    st.info("Overview uses a sample for speed. Use Real-time Detection for per-transaction analysis.")


    # -----------------------------
    # 1) RANDOM
    # -----------------------------
    if input_method == "🎲 Random from Default Dataset":
        left, right = st.columns([2, 1])

        if "rand_idx" not in st.session_state:
            st.session_state.rand_idx = None
            st.session_state.rand_proba = None
            st.session_state.rand_pred = None
            st.session_state.rand_ms = None

        with left:
            st.markdown("### 🎲 Random Transaction")

            if st.button("Analyze Random Transaction", type="primary", use_container_width=True):
                idx = int(np.random.randint(0, len(default_df)))
                row_df = default_df.iloc[[idx]]

                proba, pred, elapsed_ms = predict_one_row(row_df, threshold)

                st.session_state.rand_idx = idx
                st.session_state.rand_proba = proba
                st.session_state.rand_pred = pred
                st.session_state.rand_ms = elapsed_ms

            if st.session_state.rand_idx is not None:
                idx = st.session_state.rand_idx
                row_df = default_df.iloc[[idx]]

                # Show row WITHOUT label
                shown_row, label_col = drop_label_cols(row_df)
                st.caption(f"Selected row index: {idx}")
                st.dataframe(shown_row, use_container_width=True)

                # Show Analysis panel (your requested format)
                show_analysis_panel(
                    row_df=row_df,
                    proba=float(st.session_state.rand_proba),
                    pred=int(st.session_state.rand_pred),
                    elapsed_ms=float(st.session_state.rand_ms),
                    threshold=threshold
                )

                # Verification (raw row)
                st.markdown("### ✅ Verification")
                if label_col is not None:
                    actual = int(row_df.iloc[0][label_col])
                    st.info(f"**Actual label ({label_col})** = `{actual}` | **Predicted** = `{int(st.session_state.rand_pred)}`")
                else:
                    st.warning("No label column found. Cannot compare actual vs predicted.")
                st.dataframe(row_df, use_container_width=True)

        with right:
            st.markdown("### ℹ️ Tips")
            st.write("- Threshold 0.01 means 1.0% risk cutoff.")
            st.write("- Fraud Probability is model output.")
            st.write("- Risk Level gauge = probability × 100")

    # -----------------------------
    # 2) STREAM BY ROWS  (WITH ANALYSIS PANEL)
    # -----------------------------
    elif input_method == "📌 By Rows (Stream)":
        st.markdown("### 📌 Stream by Row Range")

        c1, c2, c3 = st.columns(3)
        with c1:
            start_row = st.number_input("Start row (0-based)", 0, len(default_df) - 1, 0)
        with c2:
            max_rows = st.number_input("Rows to stream", 1, min(5000, len(default_df)), min(200, len(default_df)))
        with c3:
            st.write("")
            stream_btn = st.button("▶️ Start Streaming", type="primary", use_container_width=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_total = metric_col1.empty()
        metric_fraud = metric_col2.empty()
        metric_legit = metric_col3.empty()

        status = st.empty()
        alert_box = st.empty()
        table_box = st.empty()

        # Live analysis placeholders
        analysis_container = st.container()

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
                proba, pred, elapsed_ms = predict_one_row(row_df, threshold)

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

                # ✅ Live Analysis for the current streamed row
                with analysis_container:
                    st.markdown("---")
                    st.caption(f"Current row: {i}")
                    show_analysis_panel(
                        row_df=row_df,
                        proba=proba,
                        pred=pred,
                        elapsed_ms=elapsed_ms,
                        threshold=threshold
                    )

                time.sleep(speed)

            st.success("✅ Streaming finished.")

    # -----------------------------
    # 3) UPLOAD DATASET
    # -----------------------------
    else:
        st.markdown("### 📤 Upload Another Dataset (CSV)")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset loaded ✅ Rows: {len(new_df)}")

            lbl = find_label_col(new_df)
            show_df = new_df.drop(columns=[lbl]) if lbl else new_df
            st.dataframe(show_df.head(20), use_container_width=True)

            if st.button("Run Prediction on Uploaded Dataset", type="primary", use_container_width=True):
                proba, pred = predict_proba_for_df(new_df, threshold)
                result = build_result_df(show_df, proba, pred)

                st.warning(f"🚨 Total Fraud Detected: {int((pred == 1).sum()):,} / {len(result):,}")
                st.dataframe(result.head(100), use_container_width=True)

                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

# ---------- Data Preview ----------
with st.expander("📄 View default dataset preview (fraudTest)"):
    st.dataframe(default_df.head(30), use_container_width=True)













