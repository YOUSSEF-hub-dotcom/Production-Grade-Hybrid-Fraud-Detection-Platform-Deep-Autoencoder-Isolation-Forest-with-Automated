import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page Configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Credit Card Fraud Detection System")

# --- INITIALIZE SESSION STATE ---
# Crucial to safeguard continuous values during form submission reruns
if 'sample_loaded' not in st.session_state:
    st.session_state['sample_loaded'] = False
for i in range(30):
    if f"feat_{i}" not in st.session_state:
        st.session_state[f"feat_{i}"] = 0.0

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
        return sample.iloc[0, :30].tolist(), int(sample.iloc[0]['Class'])
    except Exception as e:
        st.error(f"Error loading local CSV dataset: {e}")
        return [0.0] * 30, None

# --- Page 1: Real-time Prediction ---
if page == "Real-time Prediction":
    st.header("🔍 Real-time Transaction Analysis")

    col_load, col_status = st.columns([1, 4])
    
    with col_load:
        if st.button("📥 Load Random Sample", use_container_width=True):
            sample_features, true_class = get_sample_data()
            for i in range(30):
                st.session_state[f"feat_{i}"] = float(sample_features[i])
            st.session_state['true_class'] = true_class
            st.session_state['sample_loaded'] = True
            st.rerun()

    with col_status:
        if st.session_state.get('sample_loaded') and 'true_class' in st.session_state:
            label = "FRAUD" if st.session_state['true_class'] == 1 else "NORMAL"
            color = "red" if label == "FRAUD" else "green"
            st.markdown(f"<h4 style='margin-top:5px;'>Loaded Sample Ground Truth: <span style='color:{color};'>{label}</span></h4>", unsafe_allow_html=True)

    # Transaction Input Form
    with st.form("prediction_form"):
        st.info("💡 Input or audit the 30 features of the transaction to execute real-time model ensembling:")
        cols = st.columns(5)
        features = []
        
        # FIXED: Bind value to the pre-initialized session state explicitly to persist data safely
        for i in range(30):
            val = cols[i % 5].number_input(
                f"V{i + 1}", 
                format="%.4f", 
                value=st.session_state[f"feat_{i}"],
                key=f"input_feat_{i}"
            )
            features.append(val)
            
        submit = st.form_submit_button("Analyze Transaction", use_container_width=True)

    if submit:
        try:
            # Call Backend API
            with st.spinner("Analyzing parameters through deep network pipeline..."):
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
                    title={'text': "Isolation Anomaly Score", 'font': {'size': 16}},
                    gauge={'axis': {'range': [-0.5, 0.5]},
                           'bar': {'color': "#1E3A8A"},
                           'steps': [{'range': [-0.5, 0], 'color': "#FFCDD2"}, 
                                     {'range': [0, 0.5], 'color': "#C8E6C9"}]}))
                fig_iso.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
                c1.plotly_chart(fig_iso, use_container_width=True)

                # Autoencoder MSE Gauge
                fig_ae = go.Figure(go.Indicator(
                    mode="gauge+number", value=result["reconstruction_error"],
                    title={'text': "Reconstruction Error (MSE)", 'font': {'size': 16}},
                    gauge={'axis': {'range': [0, 200]},
                           'bar': {'color': "#1E3A8A"},
                           'steps': [{'range': [0, 50], 'color': "#C8E6C9"},
                                     {'range': [50, 200], 'color': "#FFE0B2"}]}))
                fig_ae.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
                c2.plotly_chart(fig_ae, use_container_width=True)

                # Latency Card Conversion
                c3.markdown("<br><br>", unsafe_allow_html=True)
                c3.metric(label="System Response Latency", value=f"{result['process_time_ms']} ms", delta="Fast Inference")

                # Radar Chart for Feature Fingerprint
                st.subheader("🕸️ Transaction Signature (Top 10 Features)")
                radar_data = pd.DataFrame(dict(r=features[:10], theta=[f"V{i + 1}" for i in range(10)]))
                fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
                fig_radar.update_traces(fill='toself', line_color='#1E3A8A')
                fig_radar.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_radar, use_container_width=True)

            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error occurred')}")
        except Exception as e:
            st.error(f"Backend Connection Failed: {str(e)}")

# --- Page 2: Historical Logs Analysis ---
elif page == "Historical Logs Analysis":
    st.header("📊 Fraud Analytics & Intelligence")
    
    # FIXED: Automatically fetch logs on initial load instead of showing a blank screen
    logs = None
    try:
        response = requests.get(f"{API_URL}/logs/fraud-only")
        if response.status_code == 200:
            logs = response.json()
    except Exception as e:
        st.error(f"Failed to auto-fetch logs from API: {str(e)}")

    if st.button("🔄 Force Refresh Database"):
        st.rerun()
        
    st.divider()

    if logs:
        df_logs = pd.DataFrame(logs)

        # Summary Metrics Cards
        m1, m2 = st.columns(2)
        m1.metric("Total Fraudulent Hits Captured", len(df_logs))
        m2.metric("Avg Severity Anomaly (MSE)", round(df_logs['mse_error'].mean(), 2))

        # Logs Table
        st.write("### 📋 Historical Fraud Log Stream (Real-Time Auditing)")
        # Dropping input_data to maintain UI neatness
        display_df = df_logs.drop(columns=['input_data']) if 'input_data' in df_logs.columns else df_logs
        st.dataframe(display_df, use_container_width=True)

        # Time Series Trend Analysis
        st.write("### 📈 Fraud Intensity Trends")
        df_logs['created_at'] = pd.to_datetime(df_logs['created_at'])
        fig_trend = px.scatter(df_logs, x="created_at", y="mse_error",
                               size="process_time_ms", color="mse_error",
                               color_continuous_scale="Reds",
                               title="Detected Anomalies Over Time (Bubble size represents processing latency)",
                               labels={"mse_error": "Severity (MSE)", "created_at": "Detection Time"})
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No fraudulent activities currently logged in the database. System is completely secure.")
