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

st.set_page_config(
    page_title="ChurnLens · E-Commerce Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
}

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Main content padding */
.main .block-container {
    padding: 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ── HEADER ── */
.churn-header {
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.churn-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.churn-header h1 span {
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.churn-header p {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.01em;
}

/* ── SIDEBAR LABEL ── */
.sidebar-section {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.25);
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.5rem;
    letter-spacing: -0.01em;
}

/* ── INPUT STYLING ── */
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.55) !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] select:focus {
    border-color: rgba(167, 139, 250, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1) !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #a78bfa !important;
}

/* ── PREDICT BUTTON ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    height: 3.2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(167, 139, 250, 0.35) !important;
}

/* ── STAT CARDS ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    transition: border-color 0.2s;
}

.stat-card:hover { border-color: rgba(255,255,255,0.14); }

.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin-bottom: 0.5rem;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}

.stat-unit {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.3);
    margin-left: 4px;
}

/* ── RESULT PANEL ── */
.result-panel {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-panel.high {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
    border: 1px solid rgba(239,68,68,0.25);
}

.result-panel.low {
    background: linear-gradient(135deg, rgba(52,211,153,0.12) 0%, rgba(16,185,129,0.06) 100%);
    border: 1px solid rgba(52,211,153,0.25);
}

.result-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

.result-badge.high {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
}

.result-badge.low {
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
}

.result-prob {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.result-prob.high { color: #f87171; }
.result-prob.low { color: #34d399; }

.result-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.35);
    margin-bottom: 1.5rem;
}

/* Progress bar */
.prob-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 6px;
    width: 100%;
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

.prob-bar-fill.high { background: linear-gradient(90deg, #ef4444, #f87171); }
.prob-bar-fill.low  { background: linear-gradient(90deg, #10b981, #34d399); }

.result-action {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 400;
    color: rgba(255,255,255,0.5);
    font-style: italic;
}

/* ── IDLE STATE ── */
.idle-panel {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
}

.idle-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}

.idle-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.2);
    font-weight: 300;
}

/* ── ABOUT EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.85rem !important;
}

[data-testid="stExpander"] p {
    font-family: 'DM Sans', sans-serif !important;
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.85rem !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1.5rem 0 !important;
}

/* Column gap */
[data-testid="column"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ──
@st.cache_resource
def load_model():
    return joblib.load('churn_pipeline.pkl')

model = load_model()

# ── HEADER ──
st.markdown("""
<div class="churn-header">
    <h1>Churn<span>Lens</span></h1>
    <p>E-Commerce Customer Intelligence · Real-time churn risk assessment</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔮 Customer Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Engagement</div>', unsafe_allow_html=True)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)
    hour_spend = st.slider("Hours on App", 0, 10, 3)
    order_count = st.number_input("Total Orders", 1, 100, 3)
    last_order = st.number_input("Days Since Last Order", 0, 100, 5)
    coupons = st.number_input("Coupons Used", 0, 50, 1)

    st.markdown('<div class="sidebar-section">Financials</div>', unsafe_allow_html=True)
    cashback = st.number_input("Cashback Amount (₹)", 0.0, 500.0, 150.0)
    hike = st.number_input("Order Amount Hike (%)", 0, 100, 15)
    warehouse_dist = st.number_input("Warehouse Distance (km)", min_value=0, max_value=200, value=15)

    st.markdown('<div class="sidebar-section">Profile</div>', unsafe_allow_html=True)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    devices = st.number_input("Devices Registered", 1, 10, 3)
    addresses = st.number_input("No. of Addresses", 1, 20, 3)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    complain = st.selectbox("Filed Complaint?", ["No", "Yes"])

    st.markdown('<div class="sidebar-section">Demographics</div>', unsafe_allow_html=True)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    login_device = st.selectbox("Login Device", ["Mobile Phone", "Phone", "Computer"])
    payment = st.selectbox("Payment Mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "COD", "CC", "Cash on Delivery"])
    order_cat = st.selectbox("Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Analyse Risk")

# ── MAIN ──
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    # Stat cards
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Tenure</div>
            <div class="stat-value">{tenure}<span class="stat-unit">mo</span></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Satisfaction</div>
            <div class="stat-value">{satisfaction}<span class="stat-unit">/ 5</span></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Orders</div>
            <div class="stat-value">{order_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prediction result
    if predict_btn:
        data = {
            'Tenure': tenure, 'CityTier': city_tier, 'WarehouseToHome': warehouse_dist,
            'HourSpendOnApp': hour_spend, 'NumberOfDeviceRegistered': devices,
            'SatisfactionScore': satisfaction, 'NumberOfAddress': addresses,
            'Complain': 1 if complain == "Yes" else 0,
            'OrderAmountHikeFromlastYear': hike, 'CouponUsed': coupons,
            'OrderCount': order_count, 'DaySinceLastOrder': last_order,
            'CashbackAmount': cashback,
            'PreferredLoginDevice_Mobile Phone': 1 if login_device == "Mobile Phone" else 0,
            'PreferredLoginDevice_Phone': 1 if login_device == "Phone" else 0,
            'PreferredPaymentMode_COD': 1 if payment == "COD" else 0,
            'PreferredPaymentMode_Cash on Delivery': 1 if payment == "Cash on Delivery" else 0,
            'PreferredPaymentMode_Credit Card': 1 if payment == "Credit Card" else 0,
            'PreferredPaymentMode_Debit Card': 1 if payment == "Debit Card" else 0,
            'PreferredPaymentMode_E wallet': 1 if payment == "E wallet" else 0,
            'PreferredPaymentMode_UPI': 1 if payment == "UPI" else 0,
            'Gender_Male': 1 if gender == "Male" else 0,
            'PreferedOrderCat_Grocery': 1 if order_cat == "Grocery" else 0,
            'PreferedOrderCat_Laptop & Accessory': 1 if order_cat == "Laptop & Accessory" else 0,
            'PreferedOrderCat_Mobile': 1 if order_cat == "Mobile" else 0,
            'PreferedOrderCat_Mobile Phone': 1 if order_cat == "Mobile Phone" else 0,
            'PreferedOrderCat_Others': 1 if order_cat == "Others" else 0,
            'MaritalStatus_Married': 1 if marital == "Married" else 0,
            'MaritalStatus_Single': 1 if marital == "Single" else 0,
        }

        input_cols = ['Tenure','CityTier','WarehouseToHome','HourSpendOnApp','NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount','DaySinceLastOrder','CashbackAmount','PreferredLoginDevice_Mobile Phone','PreferredLoginDevice_Phone','PreferredPaymentMode_COD','PreferredPaymentMode_Cash on Delivery','PreferredPaymentMode_Credit Card','PreferredPaymentMode_Debit Card','PreferredPaymentMode_E wallet','PreferredPaymentMode_UPI','Gender_Male','PreferedOrderCat_Grocery','PreferedOrderCat_Laptop & Accessory','PreferedOrderCat_Mobile','PreferedOrderCat_Mobile Phone','PreferedOrderCat_Others','MaritalStatus_Married','MaritalStatus_Single']
        input_df = pd.DataFrame([data], columns=input_cols)

        prob = model.predict_proba(input_df)[0][1]
        risk = "high" if prob > 0.5 else "low"
        risk_label = "High Risk" if risk == "high" else "Low Risk"
        action = "Consider offering a personalised retention discount or loyalty reward." if risk == "high" else "Consistent experience is working — keep it up."

        st.markdown(f"""
        <div class="result-panel {risk}">
            <div class="result-badge {risk}">{risk_label}</div>
            <div class="result-prob {risk}">{prob*100:.1f}%</div>
            <div class="result-sub">churn probability</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill {risk}" style="width: {prob*100}%;"></div>
            </div>
            <div class="result-action">{action}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="idle-panel">
            <div class="idle-icon">🔮</div>
            <div class="idle-text">Configure the customer profile<br>and click Analyse Risk</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.expander("About the Model", expanded=False):
        st.write("XGBoost classifier trained on 5,600+ e-commerce customer interactions. Achieves 96.6% accuracy identifying potential churn using behavioural, transactional, and demographic signals.")