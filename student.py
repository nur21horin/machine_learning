import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("student-mat.csv", sep=";")

# ---------------------------
# 2. Create Target Variable
# ---------------------------
df["target"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Drop original grade columns
df.drop(["G1", "G2", "G3"], axis=1, inplace=True)

# ---------------------------
# 3. Encode ALL Categorical Columns
# ---------------------------
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ---------------------------
# 4. Split Features & Target
# ---------------------------
X = df.drop("target", axis=1)
y = df["target"]

# ---------------------------
# 5. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 6. Feature Scaling (AFTER SPLIT)
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 7. Train Models
# ---------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# SVM
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# ---------------------------
# 8. Evaluation
# ---------------------------
print("\n=== ACCURACY SCORES ===")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("SVM:", accuracy_score(y_test, y_pred_svm))

print("\n=== CLASSIFICATION REPORT (Random Forest) ===")
print(classification_report(y_test, y_pred_rf))

# ---------------------------
# 9. Cross Validation (Research Level)
# ---------------------------
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\nCross Validation Accuracy:", cv_scores.mean())

# ---------------------------
# 10. Hyperparameter Tuning
# ---------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred_opt = best_model.predict(X_test)

print("\nOptimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_opt))

# ---------------------------
# 11. Confusion Matrix
# ---------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_opt))

# ---------------------------
# 12. Feature Importance (Research Insight)
# ---------------------------
importance = best_model.feature_importances_

plt.figure(figsize=(10,5))
plt.barh(X.columns, importances)
plt.xlabel("Feature Importance")
plt.title("Student Performance Feature Impact")
plt.show()