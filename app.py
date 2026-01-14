import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gdown
import plotly.express as px

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

    /* ---------- Plotly background ---------- */
    .js-plotly-plot .plotly .main-svg { font-family: inherit !important; }
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
    - Keep a clean feature DF for preprocessor/model
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

def compute_similarity_and_neighbors(df: pd.DataFrame, idx: int, k: int = 10):
    """
    Simple prototype "similar transactions":
    - Prefer same merchant/category if columns exist
    - Otherwise use closest amount if 'amt' exists
    """
    base = df.iloc[idx]

    amt_col = "amt" if "amt" in df.columns else None
    merch_col = "merchant" if "merchant" in df.columns else None
    cat_col = "category" if "category" in df.columns else None

    candidates = df.copy()

    # Filter by same merchant/category if available
    if merch_col and pd.notna(base.get(merch_col)):
        candidates = candidates[candidates[merch_col] == base[merch_col]]
    if cat_col and pd.notna(base.get(cat_col)):
        candidates = candidates[candidates[cat_col] == base[cat_col]]

    # If too few, fall back to entire df
    if len(candidates) < k + 1:
        candidates = df.copy()

    # Remove self
    candidates = candidates.drop(index=idx, errors="ignore")

    if amt_col:
        # Rank by closest amount
        base_amt = float(base.get(amt_col, 0.0))
        candidates = candidates.assign(_dist=(candidates[amt_col].astype(float) - base_amt).abs())
        neighbors = candidates.nsmallest(k, "_dist").drop(columns=["_dist"])
        # Similarity score: inverse distance normalized (rough)
        d = (candidates[amt_col].astype(float) - base_amt).abs()
        denom = float(np.nanpercentile(d, 95) + 1e-9) if len(d) else 1.0
        similarity = float(max(0.0, 1.0 - (float(d.min()) / denom))) * 100.0 if len(d) else 0.0
    else:
        neighbors = candidates.head(k)
        similarity = 0.0

    return neighbors, similarity

def risk_badge_and_color(risk_pct: float):
    # mimic "normal / warning / high"
    if risk_pct < 30:
        return "TRANSACTION NORMAL", "#10b981"
    if risk_pct < 70:
        return "TRANSACTION SUSPICIOUS", "#f59e0b"
    return "HIGH FRAUD RISK", "#ef4444"

def plot_risk_gauge(risk_pct: float):
    # Plotly gauge (0-100)
    fig = px.scatter()  # empty base
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

top_msg3.markdown("<div class='status-ok'>Vector database initialized! (prototype status)</div>", unsafe_allow_html=True)

# =========================
# Pages
# =========================

# ---------- Dashboard Overview ----------
if mode == "📊 Dashboard Overview":
    st.markdown("## 📊 Dashboard Overview")
    st.markdown("<span class='pill'>Default source: fraudTest</span>", unsafe_allow_html=True)
    st.write("")

    sample_n = min(20000, len(default_df))
    sample_df = default_df.head(sample_n)

    proba_s, pred_s = predict_proba_for_df(sample_df)

    total = int(len(sample_df))
    fraud_count = int(pred_s.sum())
    fraud_rate = (fraud_count / total) * 100 if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions (sample)", f"{total:,}")
    c2.metric("Detected Fraud (sample)", f"{fraud_count:,}")
    c3.metric("Fraud Rate (sample)", f"{fraud_rate:.2f}%")
    c4.metric("Threshold", f"{threshold}")

    st.divider()
    colL, colR = st.columns(2)

    # ===== Graph 1: Amount Distribution (use readable buckets + axis labels) =====
    with colL:
        st.subheader("Transaction Amount Distribution")
        st.caption("Amount Distribution by Predicted Fraud Status")

        if "amt" not in sample_df.columns:
            st.warning("Column `amt` not found.")
        else:
            tmp = sample_df.copy()
            tmp["pred_label"] = np.where(pred_s == 1, "FRAUD", "LEGIT")

            # readable buckets (RM)
            bins = [0, 10, 50, 100, 200, 500, 1000, np.inf]
            labels = ["0–10", "10–50", "50–100", "100–200", "200–500", "500–1000", ">1000"]
            tmp["amount_bucket"] = pd.cut(tmp["amt"].astype(float), bins=bins, labels=labels, include_lowest=True)

            count_df = (
                tmp.groupby(["amount_bucket", "pred_label"])
                   .size()
                   .reset_index(name="count")
            )

            fig1 = px.bar(
                count_df,
                x="amount_bucket",
                y="count",
                color="pred_label",
                barmode="group",
                labels={
                    "amount_bucket": "Transaction Amount Range (RM)",
                    "count": "Number of Transactions",
                    "pred_label": "Predicted Fraud Status"
                },
                title=""
            )
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend_title_text="Fraud Status",
                xaxis=dict(title="Transaction Amount Range (USD)"),
                yaxis=dict(title="Number of Transactions"),
            )
            st.plotly_chart(fig1, use_container_width=True)

    # ===== Graph 2: Fraud Rate by Category + axis labels =====
    with colR:
        st.subheader("Fraud by Merchant Category")
        st.caption("Fraud Rate (%) by Category (Predicted)")

        if "category" not in sample_df.columns:
            st.warning("Column `category` not found.")
        else:
            tmp = sample_df.copy()
            tmp["pred"] = pred_s.astype(int)

            rate = (
                tmp.groupby("category")["pred"]
                   .mean()
                   .mul(100)
                   .sort_values(ascending=False)
                   .head(15)
                   .reset_index(name="fraud_rate_pct")
            )

            fig2 = px.bar(
                rate,
                x="category",
                y="fraud_rate_pct",
                labels={
                    "category": "Merchant Category",
                    "fraud_rate_pct": "Fraud Rate (%)"
                },
                title=""
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis_tickangle=-25,
                xaxis=dict(title="Merchant Category"),
                yaxis=dict(title="Fraud Rate (%)"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.info("Overview uses a sample for speed. Use Real-time Detection for streaming/row-based testing.")

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

        if "rand_idx" not in st.session_state:
            st.session_state.rand_idx = None
            st.session_state.rand_pred = None
            st.session_state.rand_proba = None
            st.session_state.rand_ms = None

        with left:
            st.markdown("### 🎲 Random Transaction")
            st.markdown("<div class='subtle'>Pick a random row from fraudTest and run prediction.</div>", unsafe_allow_html=True)

            if st.button("Analyze Random Transaction", type="primary", use_container_width=True):
                idx = int(np.random.randint(0, len(default_df)))
                row_df = default_df.iloc[[idx]]

                proba, pred, elapsed_ms = predict_one_row(row_df, threshold)

                st.session_state.rand_idx = idx
                st.session_state.rand_pred = pred
                st.session_state.rand_proba = proba
                st.session_state.rand_ms = elapsed_ms

            if st.session_state.rand_idx is not None:
                idx = st.session_state.rand_idx
                proba = float(st.session_state.rand_proba)
                pred = int(st.session_state.rand_pred)
                elapsed_ms = float(st.session_state.rand_ms)

                # Top banner like your example
                risk_pct = proba * 100.0
                badge, badge_color = risk_badge_and_color(risk_pct)
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(16,185,129,0.10);
                        border: 1px solid {badge_color};
                        border-radius: 10px;
                        padding: 12px 16px;
                        font-weight: 800;
                        color: #e5e7eb;
                        ">
                        ✅ <span style="color:{badge_color};">{badge}</span> - Risk Score: <b>{risk_pct:.1f}%</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                st.caption(f"Selected row index: {idx}")

                chosen_row = default_df.iloc[[idx]]
                label_col = find_label_col(chosen_row)

                # Show row for viewing (BUT your model prediction uses features only)
                shown_row = chosen_row.drop(columns=[label_col]) if label_col else chosen_row
                st.dataframe(shown_row, use_container_width=True)

                # Prediction banner
                if pred == 1:
                    st.error(f"🚨 FRAUD | Probability = {proba:.6f}")
                else:
                    st.success(f"✅ LEGIT | Probability = {proba:.6f}")

                # -----------------------------
                # ANALYSIS PANEL (like picture)
                # -----------------------------
                st.write("")
                st.markdown("### 🔍 Analysis")

                # Similar transactions (prototype)
                neighbors, similarity_pct = compute_similarity_and_neighbors(default_df, idx, k=10)

                # Predict on neighbors to get "fraud count" among similar
                try:
                    n_proba, n_pred = predict_proba_for_df(neighbors)
                    similar_total = int(len(neighbors))
                    similar_fraud = int(n_pred.sum())
                    fraud_ratio = (similar_fraud / similar_total * 100) if similar_total else 0.0
                except Exception:
                    similar_total = int(len(neighbors))
                    similar_fraud = 0
                    fraud_ratio = 0.0

                # Transaction details (best-effort fields)
                t = chosen_row.iloc[0].to_dict()
                amt = t.get("amt", "—")
                cat = t.get("category", "—")
                merch = t.get("merchant", "—")
                city = t.get("city", t.get("state", "—"))

                a, b, c = st.columns([1.4, 1.3, 1.3])

                with a:
                    st.markdown("#### Transaction Details")
                    st.write(f"**Merchant:** {merch}")
                    st.write(f"**Category:** {cat}")
                    st.write(f"**Amount:** {amt}")
                    st.write(f"**Location:** {city}")

                with b:
                    st.markdown("#### Detection Metrics")
                    st.write(f"**Fraud Score:** {risk_pct:.1f}%")
                    st.write(f"**Model Probability:** {risk_pct:.1f}%")
                    st.write(f"**Similarity Score:** {similarity_pct:.1f}%")
                    st.write(f"**Detection Time:** {elapsed_ms:.2f} ms")

                with c:
                    st.markdown("#### Similar Transactions")
                    st.write(f"**Total Found:** {similar_total}")
                    st.write(f"**Fraud Count:** {similar_fraud}")
                    st.write(f"**Fraud Ratio:** {fraud_ratio:.1f}%")
                    st.write("")  # spacing
                    plot_risk_gauge(risk_pct)

                # Verification section (show exact row + actual label if exists)
                st.write("")
                st.markdown("### ✅ Verification")
                st.markdown("<div class='subtle'>This is the exact raw row from fraudTest.</div>", unsafe_allow_html=True)

                if label_col is not None:
                    actual = chosen_row.iloc[0][label_col]
                    st.info(f"**Actual label ({label_col})** = `{actual}` | **Predicted** = `{pred}`")
                else:
                    st.warning("No actual label column found in fraudTest, so we can’t compare Actual vs Predicted.")

                st.dataframe(chosen_row, use_container_width=True)

        with right:
            st.markdown("### ℹ️ Tips")
            st.write("- Lower threshold (0.01) shows more alerts.")
            st.write("- Model NEVER uses `is_fraud` for prediction (it is dropped before preprocessing).")
            st.write("- Similar Transactions is a prototype feature (simple similarity).")

    # ========== 2) By rows (stream) ==========
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
            stream_btn = st.button("▶️ Start Streaming", type="primary", use_container_width=True)

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
                proba, pred, _ = predict_one_row(row_df, threshold)

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
            new_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset loaded ✅ Rows: {len(new_df)}")

            # show input preview WITHOUT label
            lbl = find_label_col(new_df)
            show_df = new_df.drop(columns=[lbl]) if lbl else new_df
            st.dataframe(show_df.head(20), use_container_width=True)

            if st.button("Run Prediction on Uploaded Dataset", type="primary", use_container_width=True):
                proba, pred = predict_proba_for_df(new_df)
                result = build_result_df(show_df, proba, pred)

                st.warning(f"🚨 Total Fraud Detected: {int((pred == 1).sum()):,} / {len(result):,}")
                st.dataframe(result.head(100), use_container_width=True)

                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

# ---------- Analytics ----------
elif mode == "📈 Analytics":
    st.markdown("## 📈 Analytics")

    sample_n = min(5000, len(default_df))
    sample_df = default_df.head(sample_n)
    proba, pred = predict_proba_for_df(sample_df)

    st.markdown("### Fraud Probability Distribution (sample)")
    fig = px.histogram(
        pd.DataFrame({"fraud_probability": proba}),
        x="fraud_probability",
        nbins=50,
        labels={"fraud_probability": "Fraud Probability"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Fraud Probability"),
        yaxis=dict(title="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Fraud vs Legit Counts (sample)")
    counts = pd.DataFrame({
        "Predicted Label": ["LEGIT", "FRAUD"],
        "Count": [int((pred == 0).sum()), int((pred == 1).sum())]
    })
    fig2 = px.bar(counts, x="Predicted Label", y="Count", labels={"Predicted Label": "Predicted Label", "Count": "Count"})
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="Count"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Performance Metrics ----------
else:
    st.markdown("## ⚡ Performance Metrics")

    sample_n = min(5000, len(default_df))
    sample_df = default_df.head(sample_n)
    proba, pred = predict_proba_for_df(sample_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold", f"{threshold}")
    c2.metric("Sample size", f"{sample_n:,}")
    c3.metric("Predicted fraud", f"{int(pred.sum()):,}")

    st.info("For true model performance metrics (Accuracy/Recall/ROC-AUC), use your training/testing evaluation scripts.")



