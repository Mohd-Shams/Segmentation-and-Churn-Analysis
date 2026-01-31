import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pickled models
kmeans = joblib.load("cluster.pkl")          # Segmentation
stacking_model = joblib.load("churn_stack_model.pkl") # Churn prediction
preprocessor = joblib.load("churn_preprocessor.pkl")    # Preprocessing pipeline
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Segmentation & Churn", page_icon="📊", layout="wide")

# App title
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Customer Segmentation & Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input section in columns
st.header("Enter Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    avg_weekly_usage_hours = st.number_input("Avg Weekly Usage Hours", 0, 100, 10)
    last_login_days_ago = st.number_input("Last Login Days Ago", 0, 365, 15)
    support_tickets = st.number_input("Support Tickets", 0, 20, 1)

with col2:
    payment_failures = st.number_input("Payment Failures", 0, 10, 0)
    tenure_months = st.number_input("Tenure Months", 0, 60, 12)
    monthly_fee = st.number_input("Monthly Fee", 0, 2000, 400)

with col3:
    plan_type = st.selectbox("Plan Type", ['Standard', 'Premium','Basic'])

# Create DataFrame from inputs
input_df = pd.DataFrame({
    'avg_weekly_usage_hours':[avg_weekly_usage_hours],
    'last_login_days_ago':[last_login_days_ago],
    'support_tickets':[support_tickets],
    'payment_failures':[payment_failures],
    'tenure_months':[tenure_months],
    'monthly_fee':[monthly_fee],
    'plan_type':[plan_type]
})

# Prediction button
if st.button("Predict 🚀"):
    # Preprocess input for churn prediction
    input_processed = preprocessor.transform(input_df)

    # Predict churn probability
    churn_prob = stacking_model.predict_proba(input_processed)[:, 1][0]

    # Predict segment using KMeans
    cluster_features = input_df[['avg_weekly_usage_hours', 'last_login_days_ago', 
                                 'support_tickets', 'payment_failures', 
                                 'tenure_months', 'monthly_fee']]
    scaled_numeric = scaler.transform(cluster_features)
    segment_label = kmeans.predict(scaled_numeric)[0]

    # Map cluster to segment names
    segment_map = {0:'At Risk', 1:'Medium', 2:'Loyal'}
    segment_name = segment_map.get(segment_label, "Unknown")

    # Display results in columns
    st.markdown("### Prediction Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown(f"<div style='padding: 15px; background-color:#ffcccc; border-radius:10px; text-align:center;'>"
                    f"<h3>Segment</h3>"
                    f"<h2 style='color:#800000;'>{segment_name}</h2></div>", unsafe_allow_html=True)

    with res_col2:
        color = "#ff0000" if churn_prob>0.7 else "#ffa500" if churn_prob>0.4 else "#32cd32"
        st.markdown(f"<div style='padding: 15px; background-color:#f0f8ff; border-radius:10px; text-align:center;'>"
                    f"<h3>Churn Probability</h3>"
                    f"<h2 style='color:{color};'>{churn_prob:.2f}</h2></div>", unsafe_allow_html=True)

    # Suggested actions
    # Suggested actions (styled card)
    st.markdown("### Suggested Actions")

    if segment_name == 'Loyal':
        if churn_prob > 0.7:
            action_text = "⚠️ High-risk loyal customer: Offer personalized incentives or loyalty rewards."
        elif churn_prob > 0.4:
            action_text = "✅ Loyal customer: Engage with premium content or promotions to retain."
        else:
            action_text = "💡 Loyal customer: Maintain regular engagement, low risk of churn."

    elif segment_name == 'Medium':
        if churn_prob > 0.6:
            action_text = "⚠️ Medium-risk: Send reminder campaigns and check for issues."
        else:
            action_text = "💡 Medium customer: Encourage engagement to move towards loyalty."

    else:  # At Risk
        if churn_prob > 0.5:
            action_text = "🚨 At-risk: Immediate retention actions needed (discounts, follow-up calls)."
        else:
            action_text = "⚠️ At-risk: Monitor closely, may need engagement strategies."

    # Navy blue styled box
    st.markdown(
        f"""
        <div style="
            background-color:#0b1c2d;
            padding:20px;
            border-radius:12px;
            text-align:center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        ">
            <h3 style="color:#5dade2;">Recommended Action</h3>
            <p style="color:white; font-size:18px;">{action_text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


   
