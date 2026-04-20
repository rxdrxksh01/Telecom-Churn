import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

INPUT_COLUMNS = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "PreferredLoginDevice_Mobile Phone",
    "PreferredLoginDevice_Phone",
    "PreferredPaymentMode_COD",
    "PreferredPaymentMode_Cash on Delivery",
    "PreferredPaymentMode_Credit Card",
    "PreferredPaymentMode_Debit Card",
    "PreferredPaymentMode_E wallet",
    "PreferredPaymentMode_UPI",
    "Gender_Male",
    "PreferedOrderCat_Grocery",
    "PreferedOrderCat_Laptop & Accessory",
    "PreferedOrderCat_Mobile",
    "PreferedOrderCat_Mobile Phone",
    "PreferedOrderCat_Others",
    "MaritalStatus_Married",
    "MaritalStatus_Single",
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #f3f5f8;
        --surface: #ffffff;
        --surface-alt: #eef2f6;
        --border: #d9e0e8;
        --text: #132033;
        --muted: #5f6f82;
        --navy: #183b63;
        --navy-soft: #eaf1f8;
        --success: #1f7a4f;
        --success-soft: #e8f5ee;
        --danger: #b54738;
        --danger-soft: #fdecea;
        --shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #eef3f8 0%, #f8fafc 100%) !important;
        color: var(--text);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    #MainMenu, header, footer {
        visibility: hidden;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] > div {
        padding: 1.5rem 1.15rem !important;
    }

    .main .block-container {
        max-width: 1280px !important;
        padding: 2rem 2.5rem 2.5rem !important;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        color: var(--text);
    }

    .hero-card {
        background: linear-gradient(135deg, #183b63 0%, #244d7d 100%);
        border-radius: 24px;
        padding: 2rem 2.2rem;
        color: #ffffff;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }

    .hero-eyebrow {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.78;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.8rem;
        font-family: 'Manrope', sans-serif;
    }

    .hero-subtitle {
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.7;
        color: rgba(255, 255, 255, 0.82);
    }

    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.35rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: var(--text);
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
    }

    .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.84rem;
        margin-top: 0.35rem;
    }

    .result-card {
        border-radius: 24px;
        padding: 1.6rem;
        border: 1px solid var(--border);
        background: var(--surface);
    }

    .result-card.high {
        border-color: rgba(181, 71, 56, 0.24);
        background: linear-gradient(180deg, #fff7f6 0%, #ffffff 100%);
    }

    .result-card.low {
        border-color: rgba(31, 122, 79, 0.24);
        background: linear-gradient(180deg, #f5fbf7 0%, #ffffff 100%);
    }

    .result-kicker {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        color: var(--muted);
    }

    .result-status {
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .result-status.high {
        color: var(--danger);
    }

    .result-status.low {
        color: var(--success);
    }

    .result-probability {
        font-family: 'Manrope', sans-serif;
        font-size: 3.6rem;
        font-weight: 800;
        margin: 0.4rem 0;
        color: var(--text);
        line-height: 1;
    }

    .result-caption {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 1.1rem;
    }

    .progress-track {
        width: 100%;
        height: 10px;
        background: #e7edf4;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
    }

    .progress-fill.high {
        background: linear-gradient(90deg, #d16457 0%, #b54738 100%);
    }

    .progress-fill.low {
        background: linear-gradient(90deg, #2f9362 0%, #1f7a4f 100%);
    }

    .result-message {
        font-size: 0.95rem;
        line-height: 1.7;
        color: var(--text);
    }

    .info-panel {
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }

    .info-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }

    .info-value {
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .profile-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
    }

    .profile-item {
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.95rem 1rem;
    }

    .profile-key {
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .profile-value {
        color: var(--text);
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.4;
    }

    .sidebar-head {
        margin-bottom: 1.2rem;
    }

    .sidebar-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .sidebar-title {
        font-family: 'Manrope', sans-serif;
        color: var(--text);
        font-size: 1.25rem;
        font-weight: 800;
    }

    .sidebar-section {
        margin: 1.25rem 0 0.7rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 0.95rem;
        font-weight: 800;
    }

    [data-testid="stNumberInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stRadio"] label {
        color: var(--text) !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: #ffffff !important;
        color: var(--text) !important;
    }

    [data-testid="stButton"] > button {
        width: 100% !important;
        height: 3.1rem !important;
        border-radius: 14px !important;
        border: none !important;
        background: linear-gradient(135deg, #183b63 0%, #28558a 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 12px 24px rgba(24, 59, 99, 0.18) !important;
    }

    [data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #163657 0%, #234c7a 100%) !important;
    }

    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
    }

    [data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: var(--text) !important;
    }

    [data-testid="stMarkdownContainer"] p {
        line-height: 1.65;
    }

    @media (max-width: 1100px) {
        .metric-grid,
        .profile-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.2rem 1rem 2rem !important;
        }

        .hero-card {
            padding: 1.4rem;
        }

        .hero-title {
            font-size: 1.7rem;
        }

        .metric-grid,
        .profile-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


def build_input_frame(values: dict) -> pd.DataFrame:
    row = {
        "Tenure": values["tenure"],
        "CityTier": values["city_tier"],
        "WarehouseToHome": values["warehouse_dist"],
        "HourSpendOnApp": values["hour_spend"],
        "NumberOfDeviceRegistered": values["devices"],
        "SatisfactionScore": values["satisfaction"],
        "NumberOfAddress": values["addresses"],
        "Complain": 1 if values["complain"] == "Yes" else 0,
        "OrderAmountHikeFromlastYear": values["hike"],
        "CouponUsed": values["coupons"],
        "OrderCount": values["order_count"],
        "DaySinceLastOrder": values["last_order"],
        "CashbackAmount": values["cashback"],
        "PreferredLoginDevice_Mobile Phone": 1 if values["login_device"] == "Mobile Phone" else 0,
        "PreferredLoginDevice_Phone": 1 if values["login_device"] == "Phone" else 0,
        "PreferredPaymentMode_COD": 1 if values["payment"] == "COD" else 0,
        "PreferredPaymentMode_Cash on Delivery": 1 if values["payment"] == "Cash on Delivery" else 0,
        "PreferredPaymentMode_Credit Card": 1 if values["payment"] == "Credit Card" else 0,
        "PreferredPaymentMode_Debit Card": 1 if values["payment"] == "Debit Card" else 0,
        "PreferredPaymentMode_E wallet": 1 if values["payment"] == "E wallet" else 0,
        "PreferredPaymentMode_UPI": 1 if values["payment"] == "UPI" else 0,
        "Gender_Male": 1 if values["gender"] == "Male" else 0,
        "PreferedOrderCat_Grocery": 1 if values["order_cat"] == "Grocery" else 0,
        "PreferedOrderCat_Laptop & Accessory": 1 if values["order_cat"] == "Laptop & Accessory" else 0,
        "PreferedOrderCat_Mobile": 1 if values["order_cat"] == "Mobile" else 0,
        "PreferedOrderCat_Mobile Phone": 1 if values["order_cat"] == "Mobile Phone" else 0,
        "PreferedOrderCat_Others": 1 if values["order_cat"] == "Others" else 0,
        "MaritalStatus_Married": 1 if values["marital"] == "Married" else 0,
        "MaritalStatus_Single": 1 if values["marital"] == "Single" else 0,
    }
    return pd.DataFrame([row], columns=INPUT_COLUMNS)


def render_profile_item(label: str, value: str) -> str:
    return f"""
    <div class="profile-item">
        <div class="profile-key">{label}</div>
        <div class="profile-value">{value}</div>
    </div>
    """


model = load_model()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">Predictive Retention Intelligence</div>
        <div class="hero-title">Customer Churn Risk Dashboard</div>
        <div class="hero-subtitle">
            Assess the probability of customer churn using behavioral, transactional, and demographic signals.
            The interface is designed for fast review and clear decision-making.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-head">
            <div class="sidebar-eyebrow">Input Panel</div>
            <div class="sidebar-title">Customer Profile</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Engagement</div>', unsafe_allow_html=True)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)
    hour_spend = st.slider("Hours on app", 0, 10, 3)
    order_count = st.number_input("Total orders", min_value=1, max_value=100, value=3)
    last_order = st.number_input("Days since last order", min_value=0, max_value=100, value=5)
    coupons = st.number_input("Coupons used", min_value=0, max_value=50, value=1)

    st.markdown('<div class="sidebar-section">Value Signals</div>', unsafe_allow_html=True)
    cashback = st.number_input("Cashback amount", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
    hike = st.number_input("Order amount hike from last year (%)", min_value=0, max_value=100, value=15)
    warehouse_dist = st.number_input("Warehouse distance to home (km)", min_value=0, max_value=200, value=15)

    st.markdown('<div class="sidebar-section">Customer Attributes</div>', unsafe_allow_html=True)
    satisfaction = st.slider("Satisfaction score", 1, 5, 3)
    devices = st.number_input("Devices registered", min_value=1, max_value=10, value=3)
    addresses = st.number_input("Saved addresses", min_value=1, max_value=20, value=3)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    complain = st.selectbox("Complaint filed", ["No", "Yes"])

    st.markdown('<div class="sidebar-section">Demographics</div>', unsafe_allow_html=True)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    marital = st.selectbox("Marital status", ["Married", "Single", "Divorced"])
    login_device = st.selectbox("Preferred login device", ["Mobile Phone", "Phone", "Computer"])
    payment = st.selectbox(
        "Preferred payment mode",
        ["Debit Card", "Credit Card", "E wallet", "UPI", "COD", "CC", "Cash on Delivery"],
    )
    order_cat = st.selectbox(
        "Preferred order category",
        ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"],
    )

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Run churn analysis")

customer_values = {
    "tenure": tenure,
    "city_tier": city_tier,
    "warehouse_dist": warehouse_dist,
    "hour_spend": hour_spend,
    "devices": devices,
    "satisfaction": satisfaction,
    "addresses": addresses,
    "complain": complain,
    "hike": hike,
    "coupons": coupons,
    "order_count": order_count,
    "last_order": last_order,
    "cashback": cashback,
    "login_device": login_device,
    "payment": payment,
    "gender": gender,
    "order_cat": order_cat,
    "marital": marital,
}

summary_col, result_col = st.columns([1.35, 1], gap="large")

with summary_col:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Customer Snapshot</div>
            <div class="section-subtitle">
                A compact view of the most relevant operational inputs before scoring the account.
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Tenure</div>
                <div class="metric-value">{tenure} months</div>
                <div class="metric-note">Relationship duration</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Satisfaction</div>
                <div class="metric-value">{satisfaction} / 5</div>
                <div class="metric-note">Experience score</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Order Count</div>
                <div class="metric-value">{order_count}</div>
                <div class="metric-note">Historical purchase activity</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last Order</div>
                <div class="metric-value">{last_order} days</div>
                <div class="metric-note">Recency of engagement</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile_html = "".join(
        [
            render_profile_item("Preferred category", order_cat),
            render_profile_item("Payment mode", payment),
            render_profile_item("Login device", login_device),
            render_profile_item("Marital status", marital),
            render_profile_item("City tier", str(city_tier)),
            render_profile_item("Complaint status", complain),
        ]
    )

    st.markdown(f'<div class="profile-grid">{profile_html}</div></div>', unsafe_allow_html=True)

with result_col:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Prediction Output</div>
            <div class="section-subtitle">
                The model estimates churn probability from the current profile and suggests an interpretation.
            </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_btn:
        input_df = build_input_frame(customer_values)
        probability = model.predict_proba(input_df)[0][1]
        risk = "high" if probability > 0.5 else "low"
        status = "Elevated churn risk" if risk == "high" else "Lower churn risk"
        message = (
            "This profile shows signs of churn sensitivity. Consider outreach, an incentive, or a service recovery action."
            if risk == "high"
            else "This profile appears relatively stable. Maintain service quality and continue standard retention efforts."
        )

        st.markdown(
            f"""
            <div class="result-card {risk}">
                <div class="result-kicker">Model assessment</div>
                <div class="result-status {risk}">{status}</div>
                <div class="result-probability">{probability * 100:.1f}%</div>
                <div class="result-caption">Estimated probability of churn</div>
                <div class="progress-track">
                    <div class="progress-fill {risk}" style="width: {probability * 100:.1f}%"></div>
                </div>
                <div class="result-message">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-panel">
                <div class="info-label">Recommended next step</div>
                <div class="info-value">
                    Review recent engagement, complaint history, satisfaction score, and order recency to confirm whether
                    a retention intervention is needed.
                </div>
            </div>
            <div class="info-panel">
                <div class="info-label">Decision threshold</div>
                <div class="info-value">
                    Profiles above 50.0% are classified as higher churn risk by the current model configuration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-kicker">Awaiting analysis</div>
                <div class="result-status">Run the model</div>
                <div class="result-caption">
                    Complete or adjust the customer inputs in the sidebar, then run the churn analysis to see the prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Model Details", expanded=False):
    st.write(
        "The application uses a trained XGBoost pipeline built on e-commerce customer behavior, transactional activity, "
        "and profile attributes. The score should be used as a decision-support signal alongside business context."
    )
