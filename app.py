import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns

# Load model
model = joblib.load("carecost_model.pkl")

st.set_page_config(page_title="CareCost Analyzer", layout="wide")

# Sidebar options
st.sidebar.markdown(
    "<h1 style='text-align: center;'>CareCost Analyzer</h1>",
    unsafe_allow_html=True
)

st.sidebar.image("logo.png")

st.sidebar.caption(
    "CareCost is a complete Data Analysis and Machine Learning project "
    "focused on understanding and predicting medical insurance costs "
    "based on demographic and health-related features."
)

option = st.sidebar.selectbox(
    'Select one',
    ['About the Project', 'Key Observations', 'CareCost Predictor', 'Features comparison']
)

@st.cache_data
def load_data():
    df_insurance = pd.read_csv("insurance.csv")
    return df_insurance

df_insurance = load_data()


## about page of the project

if option == 'About the Project':

    st.title("🏥 CareCost Analyzer")
    st.markdown("### Medical Insurance Cost Analysis & Prediction System")

    with st.container():
        col1, col2 = st.columns([2,1])

        with col1:
            st.markdown("""
### 📌 Project Overview

**CareCost** is a machine learning web application that predicts medical insurance charges 
based on demographic and health-related attributes.

### 🎯 Project Goals
- Analyze cost-driving factors
- Build statistically valid regression models
- Deploy a production-ready ML pipeline
- Provide real-time insurance cost predictions via Streamlit
""")

        with col2:
            st.metric("Dataset Size", "1338 Records")
            st.metric("Features", "6 Inputs")
            st.metric("Model R² (Log)", "0.8047")

    st.markdown("---")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df_insurance, use_container_width=True)

    st.markdown("---")

    st.subheader("🗂 Dataset Source")
    st.info("""
Medical Cost Personal Dataset (Kaggle)

Source: Guillaume Martin  
Link: https://www.kaggle.com/datasets/mirichoi0218/insurance
""")

    st.markdown("---")

    st.subheader("🔄 Complete ML Lifecycle")
    st.markdown("""
EDA → Statistical Inference → Regularization → Residual Analysis → Deployment
""")

    
elif option == 'Key Observations':

    st.title("📊 Key Observations from EDA")

    # 1️⃣ Distribution
    st.subheader("1️⃣ Distribution of Insurance Charges")

    col1, col2 = st.columns([2,1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            df_insurance["charges"],
            bins=30,
            kde=True,
            ax=ax
        )
        ax.set_title("Distribution of Medical Charges")
        st.pyplot(fig)

    with col2:
        st.success("""
• Charges were highly right-skewed  
• Skewness ≈ 1.51  
• Log transformation reduced skewness to ≈ -0.08  
• Improved regression stability
""")

    st.markdown("---")

    # 2️⃣ Smoking Impact
    st.subheader("2️⃣ Smoking Impact")

    fig2 = px.box(
        df_insurance,
        x="smoker",
        y="charges",
        color="smoker",
        title="Smoker vs Medical Charges"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.warning("""
Smoking status is the strongest predictor of insurance cost.
Smokers incur significantly higher charges.
""")

    st.markdown("---")

    # 3️⃣ Age Trend
    st.subheader("3️⃣ Age vs Charges")

    fig3 = px.scatter(
        df_insurance,
        x="age",
        y="charges",
        color="smoker",
        trendline="ols",
        title="Age vs Charges"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info("""
Insurance charges increase with age.
Older individuals show higher medical expenditure patterns.
""")

    st.markdown("---")

    # 4️⃣ Regional Comparison
    st.subheader("4️⃣ Regional Comparison")

    fig4 = px.box(
        df_insurance,
        x="region",
        y="charges",
        color="region",
        title="Region vs Medical Charges"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.caption("Regional impact exists but is weaker compared to smoking and age.")

elif option == "CareCost Predictor":

    st.title("🤖 CareCost Insurance Cost Predictor")
    st.markdown("Enter your details below to estimate your medical insurance cost.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Information")

        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Gender", ["male", "female"])
        smoker = st.selectbox("Smoking Status", ["yes", "no"])

    with col2:
        st.subheader("🏥 Health Details")

        bmi = st.slider("BMI", 10.0, 60.0, 25.0)
        children = st.slider("Number of Children", 0, 5, 0)
        region = st.selectbox(
            "Region",
            ["southwest", "southeast", "northwest", "northeast"]
        )

    st.markdown("---")

    if st.button("🔍 Predict Insurance Cost"):

        input_data = pd.DataFrame({
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region]
        })

        prediction_log = model.predict(input_data)
        prediction = np.expm1(prediction_log)

        predicted_value = prediction[0]

        # Risk Categorization
        if predicted_value < 8000:
            risk = "🟢 Low Cost Range"
            color = "#2ecc71"
        elif predicted_value < 15000:
            risk = "🟡 Moderate Cost Range"
            color = "#f1c40f"
        else:
            risk = "🔴 High Cost Range"
            color = "#e74c3c"

        st.markdown("### 💰 Predicted Insurance Cost")

        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 12px;
                background-color: #f4f6f7;
                text-align: center;
                box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            ">
                <h1 style="color:{color};">
                    ₹ {predicted_value:,.2f}
                </h1>
                <h4>{risk}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.info(
            "Note: This prediction is based on a machine learning regression model "
            "trained on historical medical insurance data."
        )
