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

Author: Shams

Future improvements include adding visualizations, model explainability (SHAP), database integration, and enhanced UI themes.
