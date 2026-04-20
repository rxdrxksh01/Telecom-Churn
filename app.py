import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

# Set page config
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('churn_pipeline.pkl')

model = load_model()

# Header
st.title("🛒 E-Commerce Customer Churn Analysis")
st.markdown("Predict the likelihood of a customer leaving based on their behavior and transaction history.")

# Sidebar for inputs
st.sidebar.header("Customer Profile")

with st.sidebar:
    # Numeric Inputs
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=5)
    warehouse_dist = st.number_input("Distance from Warehouse", min_value=0, max_value=200, value=15)
    hour_spend = st.slider("Hours Spent on App", 0, 10, 3)
    devices = st.number_input("Number of Devices", 1, 10, 3)
    satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
    addresses = st.number_input("Number of Addresses", 1, 20, 3)
    hike = st.number_input("Order Amount Hike (%)", 0, 100, 15)
    coupons = st.number_input("Coupons Used", 0, 50, 1)
    order_count = st.number_input("Total Orders", 1, 100, 3)
    last_order = st.number_input("Days Since Last Order", 0, 100, 5)
    cashback = st.number_input("Cashback Amount", 0.0, 500.0, 150.0)
    
    # Binary/Categorical mapping
    complain = st.selectbox("Lodged Complaint?", ["No", "Yes"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    
    # Selection Inputs
    st.sidebar.markdown("---")
    st.sidebar.header("Demographics & Preferences")
    gender = st.radio("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
    payment = st.selectbox("Preferred Payment Mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "COD", "CC", "Cash on Delivery"])
    order_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"])

# Main area for prediction
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Model Prediction")
    
    if st.button("Analyze Churn Risk"):
        # Prepare input data
        # We need to map raw inputs to the specific dummy columns the model was trained on
        
        # Numeric base
        data = {
            'Tenure': tenure,
            'CityTier': city_tier,
            'WarehouseToHome': warehouse_dist,
            'HourSpendOnApp': hour_spend,
            'NumberOfDeviceRegistered': devices,
            'SatisfactionScore': satisfaction,
            'NumberOfAddress': addresses,
            'Complain': 1 if complain == "Yes" else 0,
            'OrderAmountHikeFromlastYear': hike,
            'CouponUsed': coupons,
            'OrderCount': order_count,
            'DaySinceLastOrder': last_order,
            'CashbackAmount': cashback
        }
        
        # Categorical dummies (Manual mapping based on model inspection)
        # Note: Computer, CC, Female, etc. were likely the reference categories (dropped)
        
        # login_device
        data['PreferredLoginDevice_Mobile Phone'] = 1 if login_device == "Mobile Phone" else 0
        data['PreferredLoginDevice_Phone'] = 1 if login_device == "Phone" else 0
        
        # payment
        data['PreferredPaymentMode_COD'] = 1 if payment == "COD" else 0
        data['PreferredPaymentMode_Cash on Delivery'] = 1 if payment == "Cash on Delivery" else 0
        data['PreferredPaymentMode_Credit Card'] = 1 if payment == "Credit Card" else 0
        data['PreferredPaymentMode_Debit Card'] = 1 if payment == "Debit Card" else 0
        data['PreferredPaymentMode_E wallet'] = 1 if payment == "E wallet" else 0
        data['PreferredPaymentMode_UPI'] = 1 if payment == "UPI" else 0
        
        # gender
        data['Gender_Male'] = 1 if gender == "Male" else 0
        
        # order_cat
        data['PreferedOrderCat_Grocery'] = 1 if order_cat == "Grocery" else 0
        data['PreferedOrderCat_Laptop & Accessory'] = 1 if order_cat == "Laptop & Accessory" else 0
        data['PreferedOrderCat_Mobile'] = 1 if order_cat == "Mobile" else 0
        data['PreferedOrderCat_Mobile Phone'] = 1 if order_cat == "Mobile Phone" else 0
        data['PreferedOrderCat_Others'] = 1 if order_cat == "Others" else 0
        
        # marital
        data['MaritalStatus_Married'] = 1 if marital == "Married" else 0
        data['MaritalStatus_Single'] = 1 if marital == "Single" else 0
        
        # Convert to DataFrame with EXACT columns as training
        input_cols = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone', 'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery', 'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI', 'Gender_Male', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory', 'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others', 'MaritalStatus_Married', 'MaritalStatus_Single']
        
        input_df = pd.DataFrame([data], columns=input_cols)
        
        # Prediction
        prob = model.predict_proba(input_df)[0][1]
        risk_level = "High" if prob > 0.5 else "Low"
        color = "#dc3545" if risk_level == "High" else "#28a745"
        
        st.markdown(f"""
            <div class="result-card">
                <h2 style="color: {color};">{risk_level} Churn Risk</h2>
                <p>Churn Probability: <b>{prob*100:.2f}%</b></p>
                <div style="background-color: #e9ecef; border-radius: 10px; height: 1.5rem; width: 100%;">
                    <div style="background-color: {color}; height: 100%; width: {prob*100}%; border-radius: 10px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if risk_level == "High":
            st.warning("⚠️ Action Recommended: Consider offering a targeted discount or personalized reaches.")
        else:
            st.success("✅ Loyal Customer: Continue providing consistent service quality.")

with col2:
    st.subheader("Key Statistics")
    st.info(f"Avg Tenure: {tenure} months")
    st.info(f"Satisfaction: {satisfaction}/5")
    st.info(f"Total Orders: {order_count}")

# Feature importance or details
with st.expander("About the Model"):
    st.write("This model uses an XGBoost Classifier trained on a dataset of 5,600+ e-commerce customer interactions. It achieves 96.6% accuracy in identifying potential churn.")
