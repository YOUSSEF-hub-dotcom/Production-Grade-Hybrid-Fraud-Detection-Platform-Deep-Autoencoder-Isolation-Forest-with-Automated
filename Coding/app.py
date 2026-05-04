import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page Configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Credit Card Fraud Detection System")

# --- Sidebar Configuration ---
logo_url = "https://cdn-icons-png.flaticon.com/512/2092/2092663.png"
st.sidebar.image(logo_url, use_container_width=True)
st.sidebar.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Fraud Guard AI</h1>", unsafe_allow_html=True)
st.sidebar.divider()

st.sidebar.header("🕹️ Control Panel")
page = st.sidebar.radio("Navigation", ["Real-time Prediction", "Historical Logs Analysis"])

# System Status Indicators
st.sidebar.divider()
st.sidebar.subheader("System Health")
st.sidebar.success("✅ AI Models: Active")
st.sidebar.success("✅ Database: Connected")
st.sidebar.info(f"📅 Session Date: {datetime.now().strftime('%Y-%m-%d')}")

# Backend API URL
API_URL = "http://127.0.0.1:8000"

def get_sample_data():
    """Fetch a random sample from the local dataset for testing."""
    try:
        df = pd.read_csv("creditcard.csv")
        sample = df.sample(1)
        # Extract features (first 30 columns) and the ground truth class
        return sample.iloc[0, :30].tolist(), int(sample.iloc[0]['Class'])
    except Exception as e:
        st.error(f"Error loading local CSV dataset: {e}")
        return [0.0] * 30, None

# --- Page 1: Real-time Prediction ---
if page == "Real-time Prediction":
    st.header("🔍 Real-time Transaction Analysis")

    col_load, col_status = st.columns([1, 4])
    
    with col_load:
        if st.button("📥 Load Random Sample"):
            sample_features, true_class = get_sample_data()
            # Store in session state to persist values across reruns
            for i in range(30):
                st.session_state[f"feat_{i}"] = float(sample_features[i])
            st.session_state['true_class'] = true_class
            st.rerun()

    with col_status:
        if 'true_class' in st.session_state:
            label = "FRAUD" if st.session_state['true_class'] == 1 else "NORMAL"
            color = "red" if label == "FRAUD" else "green"
            st.markdown(f"**Loaded Sample Ground Truth:** :{color}[{label}]")

    # Transaction Input Form
    with st.form("prediction_form"):
        st.info("Input the 30 PCA-transformed features of the transaction:")
        cols = st.columns(5)
        features = []
        for i in range(30):
            val = cols[i % 5].number_input(f"V{i + 1}", format="%.4f", key=f"feat_{i}")
            features.append(val)
        submit = st.form_submit_button("Analyze Transaction")

    if submit:
        try:
            # Call Backend API
            response = requests.post(f"{API_URL}/predict", json={"features": features})
            if response.status_code == 200:
                result = response.json()

                # Display Result Header
                if result["fraud_prediction"] == 1:
                    st.error(f"⚠️ ALERT: Potential Fraud Detected! (Request ID: {result['id']})")
                else:
                    st.success(f"✅ Transaction is Secure (Request ID: {result['id']})")

                # Metrics and Gauges Visualization
                c1, c2, c3 = st.columns(3)

                # Isolation Forest Gauge
                fig_iso = go.Figure(go.Indicator(
                    mode="gauge+number", value=result["iso_score"],
                    title={'text': "Isolation Anomaly Score"},
                    gauge={'axis': {'range': [-0.5, 0.5]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [-0.5, 0], 'color': "red"}, 
                                     {'range': [0, 0.5], 'color': "green"}]}))
                c1.plotly_chart(fig_iso, use_container_width=True)

                # Autoencoder MSE Gauge
                fig_ae = go.Figure(go.Indicator(
                    mode="gauge+number", value=result["reconstruction_error"],
                    title={'text': "Reconstruction Error (MSE)"},
                    gauge={'axis': {'range': [0, 200]},
                           'steps': [{'range': [0, 50], 'color': "lightgreen"},
                                     {'range': [50, 200], 'color': "orange"}]}))
                c2.plotly_chart(fig_ae, use_container_width=True)

                c3.metric("API Latency", f"{result['process_time_ms']} ms")

                # Radar Chart for Feature Fingerprint
                st.subheader("🕸️ Transaction Signature (Top 10 Features)")
                radar_data = pd.DataFrame(dict(r=features[:10], theta=[f"V{i + 1}" for i in range(10)]))
                fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
                fig_radar.update_traces(fill='toself', line_color='#1E3A8A')
                st.plotly_chart(fig_radar, use_container_width=True)

            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error occurred')}")
        except Exception as e:
            st.error(f"Backend Connection Failed: {str(e)}")

# --- Page 2: Historical Logs Analysis ---
elif page == "Historical Logs Analysis":
    st.header("📊 Fraud Analytics & Intelligence")
    
    if st.button("🔄 Refresh History"):
        try:
            response = requests.get(f"{API_URL}/logs/fraud-only")
            if response.status_code == 200:
                logs = response.json()
                if logs:
                    df_logs = pd.DataFrame(logs)

                    # Summary Metrics Cards
                    m1, m2 = st.columns(2)
                    m1.metric("Total Fraudulent Hits", len(df_logs))
                    m2.metric("Avg Severity (MSE)", round(df_logs['mse_error'].mean(), 2))

                    # Logs Table
                    st.write("### 📋 Historical Fraud Log")
                    st.dataframe(df_logs.drop(columns=['input_data']), use_container_width=True)

                    # Time Series Trend Analysis
                    st.write("### 📈 Fraud Intensity Trends")
                    df_logs['created_at'] = pd.to_datetime(df_logs['created_at'])
                    fig_trend = px.scatter(df_logs, x="created_at", y="mse_error",
                                           size="process_time_ms", color="mse_error",
                                           color_continuous_scale="Reds",
                                           title="Detected Anomalies Over Time",
                                           labels={"mse_error": "Severity (MSE)", "created_at": "Detection Time"})
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning("No fraudulent activities currently logged in the database.")
        except Exception as e:
            st.error(f"Failed to fetch logs: {str(e)}")
