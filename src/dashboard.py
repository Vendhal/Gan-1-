import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "outputs", "logs", "inference_logs.csv")

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="GAN Monitoring Dashboard",
    layout="wide"
)

st.title("GAN Monitoring Dashboard")
st.caption("Module 6 – Monitoring & Performance Analysis")

# -------------------------------------------------
# Log existence check
# -------------------------------------------------
if not os.path.exists(LOG_FILE):
    st.warning("Inference logs not found. Run inference or API first.")
    st.stop()

df = pd.read_csv(LOG_FILE)

# -------------------------------------------------
# Empty log guard (THIS kills Altair warnings)
# -------------------------------------------------
if df.empty:
    st.info("Logs exist but no inference records yet.")
    st.stop()

# -------------------------------------------------
# Metrics
# -------------------------------------------------
total_requests = len(df)
failures = df["status"].str.contains("FAILED", na=False).sum()
avg_latency = round(df["latency_sec"].mean(), 4)

col1, col2, col3 = st.columns(3)

col1.metric("Total Requests", total_requests)
col2.metric("Failures", failures)
col3.metric("Avg Latency (sec)", avg_latency)

st.divider()

# -------------------------------------------------
# Latency chart (safe)
# -------------------------------------------------
st.subheader("Inference Latency Over Time")

if "latency_sec" in df.columns:
    st.line_chart(df["latency_sec"])
else:
    st.info("Latency data not available yet.")

# -------------------------------------------------
# Logs table
# -------------------------------------------------
st.subheader("Inference Logs")
st.dataframe(df, use_container_width=True)
