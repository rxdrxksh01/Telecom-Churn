import joblib
import pandas as pd
import streamlit as st
from llm_workflow import run_retention_workflow

st.set_page_config(
    page_title="Churn Studio",
    page_icon=":bar_chart:",
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

DEFAULT_INPUTS = {
    "tenure": 5,
    "hour_spend": 3,
    "order_count": 3,
    "last_order": 5,
    "coupons": 1,
    "cashback": 150.0,
    "hike": 15,
    "warehouse_dist": 15,
    "satisfaction": 3,
    "devices": 3,
    "addresses": 3,
    "city_tier": 1,
    "complain": "No",
    "gender": "Male",
    "marital": "Married",
    "login_device": "Mobile Phone",
    "payment": "Debit Card",
    "order_cat": "Laptop & Accessory",
}

HIGH_CHURN_DEMO = {
    "tenure": 1,
    "hour_spend": 1,
    "order_count": 1,
    "last_order": 30,
    "coupons": 0,
    "cashback": 20.0,
    "hike": 5,
    "warehouse_dist": 35,
    "satisfaction": 1,
    "devices": 1,
    "addresses": 1,
    "city_tier": 3,
    "complain": "Yes",
    "gender": "Female",
    "marital": "Single",
    "login_device": "Computer",
    "payment": "COD",
    "order_cat": "Others",
}

LOW_CHURN_DEMO = {
    "tenure": 18,
    "hour_spend": 7,
    "order_count": 12,
    "last_order": 2,
    "coupons": 6,
    "cashback": 220.0,
    "hike": 24,
    "warehouse_dist": 8,
    "satisfaction": 5,
    "devices": 4,
    "addresses": 4,
    "city_tier": 1,
    "complain": "No",
    "gender": "Male",
    "marital": "Married",
    "login_device": "Mobile Phone",
    "payment": "Credit Card",
    "order_cat": "Laptop & Accessory",
}

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


def init_state():
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("page", "welcome")
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("retention_output", None)
    st.session_state.setdefault("copilot_chat", [])


def apply_profile(values: dict):
    for key, value in values.items():
        st.session_state[key] = value


def collect_values() -> dict:
    return {
        "tenure": st.session_state["tenure"],
        "city_tier": st.session_state["city_tier"],
        "warehouse_dist": st.session_state["warehouse_dist"],
        "hour_spend": st.session_state["hour_spend"],
        "devices": st.session_state["devices"],
        "satisfaction": st.session_state["satisfaction"],
        "addresses": st.session_state["addresses"],
        "complain": st.session_state["complain"],
        "hike": st.session_state["hike"],
        "coupons": st.session_state["coupons"],
        "order_count": st.session_state["order_count"],
        "last_order": st.session_state["last_order"],
        "cashback": st.session_state["cashback"],
        "login_device": st.session_state["login_device"],
        "payment": st.session_state["payment"],
        "gender": st.session_state["gender"],
        "order_cat": st.session_state["order_cat"],
        "marital": st.session_state["marital"],
    }


def render_welcome():
    st.title("Churn Studio")
    st.write(
        "Welcome to the churn prediction workspace. Build a customer profile, run the model, "
        "and review churn risk with clear, business-ready outputs."
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.subheader("1) Load a profile")
            st.caption("Use default values or apply a high-risk / low-risk demo.")
    with c2:
        with st.container(border=True):
            st.subheader("2) Run prediction")
            st.caption("Generate churn probability with one click from the sidebar.")
    with c3:
        with st.container(border=True):
            st.subheader("3) Review insights")
            st.caption("Interpret score and profile snapshot before taking action.")

    st.divider()
    btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 2.0], gap="small")
    with btn_col1:
        if st.button("Open Analysis Workspace", type="primary", use_container_width=True):
            st.session_state["page"] = "analysis"
            st.rerun()
    with btn_col2:
        if st.button("Start with High-Risk Demo", use_container_width=True):
            apply_profile(HIGH_CHURN_DEMO)
            st.session_state["page"] = "analysis"
            st.rerun()


def render_sidebar(model):
    with st.sidebar:
        st.title("Profile Inputs")
        st.caption("Edit customer attributes, then run prediction.")

        b1, b2 = st.columns(2, gap="small")
        with b1:
            if st.button("High demo", use_container_width=True):
                apply_profile(HIGH_CHURN_DEMO)
                st.rerun()
        with b2:
            if st.button("Low demo", use_container_width=True):
                apply_profile(LOW_CHURN_DEMO)
                st.rerun()

        st.divider()
        st.subheader("Engagement")
        st.number_input("Tenure (months)", min_value=0, max_value=100, key="tenure", step=1)
        st.slider("Hours on app", 0, 10, key="hour_spend")
        st.number_input("Total orders", min_value=1, max_value=100, key="order_count", step=1)
        st.number_input("Days since last order", min_value=0, max_value=100, key="last_order", step=1)
        st.number_input("Coupons used", min_value=0, max_value=50, key="coupons", step=1)

        st.subheader("Value")
        st.number_input("Cashback amount", min_value=0.0, max_value=500.0, step=10.0, key="cashback")
        st.number_input("Order hike last year (%)", min_value=0, max_value=100, key="hike", step=1)
        st.number_input("Warehouse distance (km)", min_value=0, max_value=200, key="warehouse_dist", step=1)

        st.subheader("Attributes")
        st.slider("Satisfaction score", 1, 5, key="satisfaction")
        st.number_input("Devices registered", min_value=1, max_value=10, key="devices", step=1)
        st.number_input("Saved addresses", min_value=1, max_value=20, key="addresses", step=1)
        st.selectbox("City tier", [1, 2, 3], key="city_tier")
        st.selectbox("Complaint filed", ["No", "Yes"], key="complain")
        st.radio("Gender", ["Male", "Female"], horizontal=True, key="gender")
        st.selectbox("Marital status", ["Married", "Single", "Divorced"], key="marital")
        st.selectbox("Preferred device", ["Mobile Phone", "Phone", "Computer"], key="login_device")
        st.selectbox("Payment mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "COD", "CC", "Cash on Delivery"], key="payment")
        st.selectbox("Order category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"], key="order_cat")

        run = st.button("Run Churn Analysis", use_container_width=True, type="primary")
        generate = st.button("Generate LLM Retention Plan", use_container_width=True)
        back = st.button("Back to Welcome", use_container_width=True)

        if run:
            input_df = build_input_frame(collect_values())
            st.session_state["prediction"] = float(model.predict_proba(input_df)[0][1])
            st.session_state["retention_output"] = None
            st.session_state["copilot_chat"] = []
        if generate:
            input_df = build_input_frame(collect_values())
            p = float(model.predict_proba(input_df)[0][1])
            st.session_state["prediction"] = p
            with st.spinner("Running LangGraph retention workflow..."):
                st.session_state["retention_output"] = run_retention_workflow(
                    collect_values(),
                    p,
                    user_message="Generate churn-focused retention plan from model output.",
                    chat_history=st.session_state.get("copilot_chat", []),
                )
        if back:
            st.session_state["page"] = "welcome"
            st.rerun()


def render_analysis():
    values = collect_values()
    p = st.session_state["prediction"]
    retention_output = st.session_state.get("retention_output")
    st.title("Analysis Workspace")
    st.caption("Live retention risk scoring for a single customer profile.")

    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.metric("Tenure", f"{values['tenure']} mo")
    k2.metric("Satisfaction", f"{values['satisfaction']} / 5")
    k3.metric("Order Count", str(values["order_count"]))
    k4.metric("Last Order", f"{values['last_order']} days")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        tab1, tab2 = st.tabs(["Profile Snapshot", "Interpretation"])
        with tab1:
            a1, a2, a3 = st.columns(3)
            a1.metric("Order Category", values["order_cat"])
            a2.metric("Payment Mode", values["payment"])
            a3.metric("Preferred Device", values["login_device"])
            b1, b2, b3 = st.columns(3)
            b1.metric("Marital Status", values["marital"])
            b2.metric("City Tier", str(values["city_tier"]))
            b3.metric("Complaint Filed", values["complain"])
            st.dataframe(pd.DataFrame([values]), use_container_width=True, hide_index=True)
        with tab2:
            st.info(
                "Higher churn risk generally appears when tenure is short, satisfaction is low, "
                "and recent activity declines."
            )
            st.warning(
                "Use model output as a support signal with CRM history, support tickets, and campaign context."
            )

    with right:
        with st.container(border=True):
            st.subheader("Prediction")
            if p is None:
                st.caption("Run the analysis from the sidebar to generate score.")
            else:
                risk_high = p > 0.5
                score_pct = p * 100
                st.metric("Churn Probability", f"{score_pct:.1f}%")
                st.progress(int(round(score_pct)))
                if risk_high:
                    st.error("Risk Level: HIGH")
                    st.caption("Recommended: immediate retention outreach.")
                else:
                    st.success("Risk Level: LOW")
                    st.caption("Recommended: nurture and loyalty upsell.")

    st.divider()
    st.subheader("LLM Retention Copilot")
    st.caption("LangChain + LangGraph outputs with guardrails and tool-based recommendations.")
    if p is None:
        st.info("Run churn analysis first, then generate LLM retention plan from the sidebar.")
    elif retention_output is None:
        st.info("Click `Generate LLM Retention Plan` in the sidebar to produce factors, actions, and email draft.")
    else:
        if retention_output.get("error"):
            st.error(f"LLM workflow error: {retention_output['error']}")
            st.caption("Tip: set `GROQ_API_KEY` in your environment.")
        else:
            risk_level = retention_output.get("risk_level", "UNKNOWN")
            visual_fn = {
                "HIGH": st.error,
                "MEDIUM": st.warning,
                "LOW": st.success,
            }.get(risk_level, st.info)
            visual_fn(f"Detected Risk: {risk_level}")

            left_col, right_col = st.columns(2, gap="large")
            with left_col:
                st.markdown("**Why this customer is churning / staying**")
                reasons = retention_output.get("factors", [])
                if reasons:
                    for item in reasons:
                        st.write(f"- {item}")
                else:
                    st.write("- No strong model-aligned factors returned.")

                st.markdown("**How to reduce churn (or strengthen loyalty)**")
                suggestions = retention_output.get("suggestions", [])
                if suggestions:
                    for item in suggestions:
                        st.write(f"- {item}")
                else:
                    st.write("- No suggestions generated.")

                st.markdown("**Positive reasons for retention**")
                positive = retention_output.get("positive_reasons", [])
                if positive:
                    for item in positive:
                        st.write(f"- {item}")
                else:
                    st.write("- No positive reason list generated.")

            with right_col:
                st.markdown("**What to reward this user with**")
                rewards = retention_output.get("rewards", [])
                if rewards:
                    for item in rewards:
                        st.write(f"- {item}")
                else:
                    st.write("- No reward recommendations available.")

                st.markdown("**Direct actions to execute (LLM-ready tasks)**")
                actions = retention_output.get("next_actions", [])
                if actions:
                    for item in actions:
                        st.write(f"- {item}")
                else:
                    st.write("- No direct actions generated.")

            st.markdown("**Draft email to send this customer**")
            st.text_area(
                "Retention email draft",
                value=retention_output.get("email_draft", ""),
                height=220,
                key="retention_email_draft",
            )

            st.markdown("**Churn Copilot Chat (context-aware)**")
            for message in st.session_state.get("copilot_chat", []):
                speaker = "You" if message.get("role") == "user" else "Copilot"
                st.write(f"**{speaker}:** {message.get('content', '')}")

            prompt = st.text_input("Ask churn copilot", key="copilot_prompt")
            ask = st.button("Send to Copilot", use_container_width=True)
            if ask and prompt.strip():
                st.session_state["copilot_chat"].append({"role": "user", "content": prompt.strip()})
                with st.spinner("Copilot is responding with churn-only context..."):
                    response = run_retention_workflow(
                        values,
                        p,
                        user_message=prompt.strip(),
                        chat_history=st.session_state.get("copilot_chat", []),
                    )
                st.session_state["retention_output"] = response
                reply = response.get("chat_reply", "")
                if reply:
                    st.session_state["copilot_chat"].append({"role": "assistant", "content": reply})
                st.rerun()

    with st.expander("Model Details", expanded=False):
        st.write(
            "This app uses a trained XGBoost pipeline over engagement, transaction, and profile features. "
            "The output is a probability score that should be reviewed along with CRM context. "
            "For AI support, this app also uses a LangChain + LangGraph retention workflow (Groq 70B) with strict churn-only guardrails."
        )


init_state()
model = load_model()
if st.session_state["page"] == "welcome":
    render_welcome()
else:
    render_sidebar(model)
    render_analysis()
