"""Optional Streamlit UI. Every analysis is an HTTP call to the same Flask API.

Start Flask first, then: streamlit run app.py --server.port 8502
"""
import hashlib
import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("SMART_LISTENER_API_URL", "http://127.0.0.1:5000").rstrip("/")
st.set_page_config(page_title="Smart Listener", page_icon="🎧", layout="wide")
st.title("Smart Listener")
st.caption("Machine audio diagnostics · Flask API + exported PyTorch models")

with st.sidebar:
    st.subheader("Inference system")
    st.code(API_URL, language=None)
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=10)
        health = response.json()
        if response.ok:
            st.success(f"Ready · {len(health['models'])} models")
        else:
            st.error("Model system is not ready")
            st.json(health.get("errors", []))
        st.caption(f"Preprocessing: {health.get('preprocessing', {}).get('fingerprint', 'unavailable')}")
    except (requests.RequestException, ValueError):
        st.error("Flask API is offline. Start main.py first.")
        health = {}
    st.info("Auto reads fan/pump from a filename. Other filenames require a selected machine type.")

left, right = st.columns([1, 1.3], gap="large")
with left:
    st.subheader("Your recording")
    uploaded = st.file_uploader("Upload a WAV recording", type=["wav"])
    machine = st.selectbox("Machine type", ["fan", "pump", "auto"],
                           format_func=lambda item: "Auto from filename" if item == "auto" else item.title())
    snr = st.selectbox("Recording noise condition", ["6_dB", "0_dB", "-6_dB"])
    input_signature = (uploaded.name, hashlib.sha256(uploaded.getvalue()).hexdigest(), machine, snr) if uploaded else None
    if st.session_state.get("input_signature") != input_signature:
        st.session_state.pop("result", None)
        st.session_state["input_signature"] = input_signature
    if uploaded:
        st.audio(uploaded.getvalue(), format="audio/wav")
    st.caption("1–60 seconds · up to 25 MiB · the exported pipeline analyzes 10 seconds and pads shorter audio.")
    if st.button("Analyze recording", type="primary", disabled=uploaded is None or health.get("status") != "ready"):
        with st.spinner("Evaluating the exported model…"):
            try:
                response = requests.post(f"{API_URL}/api/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), "audio/wav")},
                    data={"machine_type": machine, "snr": snr}, timeout=120)
                body = response.json()
                if response.ok:
                    st.session_state["result"] = body
                else:
                    st.session_state.pop("result", None)
                    st.error(body.get("error", {}).get("message", "Analysis failed."))
            except (requests.RequestException, ValueError):
                st.session_state.pop("result", None)
                st.error("Could not reach the Flask API. Check that it is running.")

with right:
    result = st.session_state.get("result")
    st.subheader("Analysis result")
    if not result:
        st.info("Upload a recording and run an analysis to see the model's decision.")
    else:
        st.caption(f"Recording: {result['filename']}")
        (st.warning if result["is_anomaly"] else st.success)(
            f"{result['machine_type'].title()} · {result['condition'].title()}")
        st.caption(result["machine_identification"]["detail"])
        columns = st.columns(3)
        columns[0].metric("Anomaly score", f"{result['score']:.6f}")
        columns[1].metric("Decision threshold", f"{result['threshold']:.6f}")
        columns[2].metric("Processing time", f"{result['elapsed_ms']:.0f} ms")
        timeline = pd.DataFrame({"Time (s)": [w["start"] + 0.5 for w in result["window_times"]],
                                 "Window score": result["window_scores"],
                                 "Threshold": [result["threshold"]] * len(result["window_scores"])})
        st.line_chart(timeline, x="Time (s)", y=["Window score", "Threshold"])
        st.caption(f"Model: {result['model_id']} · Aggregation: {result['aggregation']} · Abnormal if score > threshold")
        for warning in result["warnings"]:
            st.warning(warning)
        st.download_button("Download JSON report", json.dumps(result, indent=2),
                           file_name="smart-listener-report.json", mime="application/json")
        with st.expander("Request / response details"):
            st.json(result)
