import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("cancer_risk_dataset_extended.csv")

X = df.drop(columns=["Cancer Risk Level"])
y = df["Cancer Risk Level"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC()
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"SVM Model Accuracy: {accuracy:.4f}")
print(report)

joblib.dump(svm_model, "svm_cancer_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("SVM model and scaler saved successfully!")
