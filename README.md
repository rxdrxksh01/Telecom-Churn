# 🛒 E-Commerce Customer Churn Prediction

An end-to-end Machine Learning solution to predict customer churn in an E-Commerce domain. This project includes data exploration, model training, and a professional Streamlit web application for real-time predictions.

## 🚀 Overview
Churn prediction helps businesses identify customers at risk of leaving. This project uses behavior analysis (tenure, satisfaction score, complaints, order history) to predict churn with **96.6% accuracy**.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, Scikit-learn, XGBoost, Streamlit, LangChain, LangGraph
- **Model:** Tuned XGBoost Classifier
- **Deployment Ready:** Configured for local and cloud hosting (Streamlit Cloud, Heroku)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd churn_prediciton
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set LLM API key (required for retention copilot):**
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
The model is trained on an E-Commerce dataset containing:
- **Demographics:** Gender, Marital Status, City Tier
- **Engagement:** Tenure, Hours Spend on App, Satisfaction Score
- **Transaction:** Order Count, Cashback, Coupons Used, Warehouse Distance
- **Support:** Complaints

## 🧠 Model Pipeline
The pipeline handles:
1. **Missing Value Imputation:** Median for numeric, most frequent for categorical.
2. **Feature Engineering:** One-Hot Encoding and dummy variable creation.
3. **Classification:** XGBoost algorithm optimized for high recall on churners.

## 🤖 LLM Retention Copilot (LangChain + LangGraph + Groq 70B)
When you click **Generate LLM Retention Plan**, the app now:
1. Uses churn probability + profile data as structured context.
2. Runs a guarded LLM step (Groq `llama3-70b-8192`) to explain churn/stability factors.
3. Keeps chat memory so follow-up prompts remember model context and conversation.
4. Returns:
   - Why this user may churn (or stay)
   - How to reduce churn / strengthen loyalty
   - Reward suggestions
   - A draft email for outreach

### Guardrails
- Uses only provided profile and model score
- Avoids certainty claims and harmful targeting
- Refuses non-churn tasks and redirects to churn-retention scope
- Produces structured, auditable output

## 📄 License
MIT License

## Feature Engineering
- Handled missing values
- Encoded categorical features
