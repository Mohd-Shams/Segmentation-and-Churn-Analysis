import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load data
df = pd.read_csv("customer_subscription_churn_usage_patterns.csv")  

# Features and target
X = df.drop(columns=['churn'])
y = df['churn'].map({'No':0, 'Yes':1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define columns
num_cols = ['avg_weekly_usage_hours','last_login_days_ago','support_tickets','payment_failures','tenure_months','monthly_fee']
cat_cols = ['plan_type']

# Preprocessor
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols),
                                               ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)])

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# Define stacking model
estimators = [('rf', RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=1,random_state=42)),
              ('gb', GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42))]

stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)

# Fit model
stack_model.fit(X_train_processed, y_train)

# Predict probabilities
y_proba = stack_model.predict_proba(X_test_processed)[:,1]

# Apply threshold
threshold = 0.4
y_pred = (y_proba >= threshold).astype(int)

# Evaluation
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# Save model and preprocessor
joblib.dump(stack_model, "churn_stack_model.pkl")
joblib.dump(preprocessor, "churn_preprocessor.pkl")
