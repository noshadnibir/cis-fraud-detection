import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# --- Configuration & Styling ---
st.set_page_config(page_title="Fraud Detection Engine", layout="wide")
st.title("🛡️ IEEE-CIS Fraud Detection Engine")
st.markdown("""
This application utilizes a gradient-boosted tree model (XGBoost) optimized for detecting anomalous financial transactions.
""")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model("model/exports/xgboost_model.json")
    with open("model/exports/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("model/exports/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, encoders, feature_names

try:
    model, encoders, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts. Error: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Transaction Parameters")
trans_amt = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
hist_mean_amt = st.sidebar.number_input("User's Historical Mean Amount ($)", min_value=0.0, value=50.0)
hist_std_amt = st.sidebar.number_input("User's Historical Std Dev ($)", min_value=0.0, value=10.0)

st.sidebar.header("Card Details")
card1 = st.sidebar.number_input("Card 1 ID", min_value=1000, max_value=20000, value=10000)
card4 = st.sidebar.selectbox("Card Network", ["visa", "mastercard", "discover", "american express"])
card6 = st.sidebar.selectbox("Card Type", ["credit", "debit"])

st.sidebar.header("Digital Identity")
p_email = st.sidebar.text_input("Purchaser Email", "user@gmail.com")
r_email = st.sidebar.text_input("Recipient Email", "user@gmail.com")
device_info = st.sidebar.text_input("Device Info", "iOS Device")

# --- Inference Pipeline ---
def preprocess_input():
    # Construct base dictionary matching the exact expected features
    input_data = {col: np.nan for col in feature_names}
    
    # 1. Direct Mappings
    input_data['TransactionAmt'] = trans_amt
    input_data['card1'] = card1
    input_data['card4'] = card4
    input_data['card6'] = card6
    
    # 2. Advanced Feature Engineering Simulation
    input_data['TransactionAmt_normalized'] = (trans_amt - hist_mean_amt) / (hist_std_amt + 1e-5)
    input_data['email_match'] = 1 if p_email == r_email else 0
    
    email_maps = {
        'gmail.com': 'google', 'gmail': 'google', 'googlemail.com': 'google',
        'hotmail.com': 'microsoft', 'outlook.com': 'microsoft', 'msn.com': 'microsoft', 'live.com': 'microsoft',
        'yahoo.com': 'yahoo', 'ymail.com': 'yahoo', 'rocketmail.com': 'yahoo',
        'icloud.com': 'apple', 'me.com': 'apple', 'mac.com': 'apple'
    }
    
    p_domain = p_email.split('@')[-1] if '@' in p_email else 'unknown'
    r_domain = r_email.split('@')[-1] if '@' in r_email else 'unknown'
    
    input_data['P_emaildomain_bin'] = email_maps.get(p_domain, 'unknown')
    input_data['R_emaildomain_bin'] = email_maps.get(r_domain, 'unknown')
    
    device_lower = device_info.lower()
    if 'ios' in device_lower or 'iphone' in device_lower: device_mapped = 'apple'
    elif 'samsung' in device_lower or 'sm-' in device_lower: device_mapped = 'samsung'
    elif 'huawei' in device_lower: device_mapped = 'huawei'
    elif 'moto' in device_lower: device_mapped = 'motorola'
    elif 'rv:' in device_lower or 'windows' in device_lower: device_mapped = 'windows'
    else: device_mapped = 'other'
    
    input_data['device_name'] = device_mapped

    df = pd.DataFrame([input_data])
    
    # 3. Label Encoding with Strict Type Checking
    for col, le in encoders.items():
        if col in df.columns:
            raw_val = df[col].iloc[0]
            
            # Evaluate for nulls before string casting
            val = 'Unknown' if pd.isna(raw_val) else str(raw_val)
            
            # Safe transform for unseen data
            if val not in le.classes_:
                val = 'Unknown_Unseen' 
                
            df[col] = le.transform([val])[0]
            
    # Ensure correct column order and data types
    df = df[feature_names]
    # Convert object columns to float to avoid XGBoost errors on missing numericals
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

# --- Execution ---
if st.button("Analyze Transaction", type="primary", use_container_width=True):
    with st.spinner('Running inference...'):
        processed_df = preprocess_input()
        
        # Predict probability
        prob = model.predict_proba(processed_df)[0][1]
        
        # Threshold determined from PR Curve Analysis in training
        OPTIMAL_THRESHOLD = 0.2323
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Fraud Probability Score", f"{prob * 100:.2f}%")
        
        with col2:
            if prob >= OPTIMAL_THRESHOLD:
                st.error("🚨 **ALERT: Transaction Flagged as Fraudulent**")
            else:
                st.success("✅ **CLEARED: Transaction Appears Legitimate**")
                
        st.markdown("### Risk Explanation")
        
        if processed_df['TransactionAmt_normalized'].iloc[0] > 3:
            st.warning("- High anomaly: Transaction amount is over 3 standard deviations above the user's historical mean.")
        if processed_df['email_match'].iloc[0] == 0:
            st.info("- Note: Purchaser and Recipient email domains do not match.")