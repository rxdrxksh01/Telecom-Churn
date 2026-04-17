# 🛒 E-Commerce Customer Churn Prediction

An end-to-end Machine Learning solution to predict customer churn in an E-Commerce domain. This project includes data exploration, model training, and a professional Streamlit web application for real-time predictions.

## 🚀 Overview
Churn prediction helps businesses identify customers at risk of leaving. This project uses behavior analysis (tenure, satisfaction score, complaints, order history) to predict churn with **96.6% accuracy**.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, Scikit-learn, XGBoost, Streamlit
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

4. **Run the application:**
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

## 📄 License
MIT License

## Feature Engineering
- Handled missing values
- Encoded categorical features
