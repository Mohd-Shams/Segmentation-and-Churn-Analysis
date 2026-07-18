📊 Customer Segmentation & Churn Prediction App

A Machine Learning powered Streamlit web application that predicts customer churn probability and classifies customers into segments (Loyal, Medium, At Risk) to support business decision-making.

Live App: https://segmentation-and-churn-analysis.streamlit.app/

This app uses a trained Stacking Classifier model for churn prediction and a KMeans clustering model for customer segmentation.
A preprocessing pipeline handles feature scaling and categorical encoding. 
The user inputs customer details such as average weekly usage hours, last login days ago, number of support tickets, payment failures, tenure months, monthly fee, and plan type.
Based on these inputs, the app displays the predicted customer segment, churn probability, and a recommended business action.

Tech stack: Python, Streamlit, Pandas, NumPy, Scikit-learn, Joblib.

How to run locally:
Clone the repository, install dependencies using pip install -r requirements.txt, and run the app with streamlit run app.py.

Output: The app shows the customer segment, churn probability score, and a recommended retention strategy.

Use cases: Customer retention analysis, ML portfolio project, business decision support, and educational ML deployment example.

📊 Power BI & SQL Analytics Extension
Business Intelligence Dashboard

To complement the Machine Learning pipeline, I developed a comprehensive Business Intelligence solution using MySQL and Microsoft Power BI. This transforms model predictions into actionable business insights for decision-makers.

Dashboard Pages
📌 Executive Dashboard

Provides a high-level overview of business performance.

KPIs

Total Customers
Total Monthly Revenue
Average Monthly Fee
Overall Churn Rate
High-Risk Customers
Average Weekly Usage

Key Visualizations

Customer Distribution
Revenue by Subscription Plan
Revenue at Risk
Churn by Customer Segment
Top High-Risk Customers
👥 Customer Segmentation Dashboard

Analyzes customer groups generated through K-Means clustering.

Analysis Includes

Customer Distribution
Revenue Contribution by Segment
Average Monthly Fee
Weekly Usage Analysis
Support Ticket Analysis
Payment Failure Analysis
⚠️ Churn Analysis Dashboard

Identifies customers likely to leave and highlights behavioral churn drivers.

Analysis Includes

Churn Rate by Customer Segment
Churn by Activity Level
Churn by Tenure
Churn by Subscription Plan
Churn Probability Distribution
Revenue at Risk
High-Risk Customer List
💰 Revenue Dashboard

Analyzes recurring revenue and customer value.

Analysis Includes

Revenue by Subscription Plan
Revenue by Customer Segment
Revenue Contribution
Top Revenue Customers
Monthly Fee Distribution
Revenue vs Support Tickets
📈 SQL Business Analysis

The dataset was imported into MySQL, where 30 business-focused SQL queries were developed to answer real-world business questions.

Topics Covered
Customer Segmentation Analysis
Revenue Analysis
Churn Analysis
Customer Behaviour Analysis
Aggregate Functions
GROUP BY & HAVING
Joins
Subqueries
Common Table Expressions (CTEs)
Window Functions
Ranking Functions
Running Totals
Business KPI Analysis
📊 Key Business Insights
Analyzed 2,800 customer records.
Identified an overall churn rate of 57.32%.
Detected 391 high-risk customers with a churn probability greater than 80%.
Generated insights from ₹12.16 Lakhs in monthly recurring revenue.
Found that customer engagement is a stronger predictor of churn than subscription plan.
Built interactive dashboards to support executive-level decision-making.




Author: Shams

Future improvements include adding visualizations, model explainability (SHAP), database integration, and enhanced UI themes.
