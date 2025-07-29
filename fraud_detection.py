import os
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -------------------------------
# STEP 1: Load All PKL Files
# -------------------------------

data_path = "./dataset"  
all_files = sorted(glob(os.path.join(data_path, "*.pkl")))

print(f"Found {len(all_files)} files.")

dataframes = [pd.read_pickle(file) for file in all_files]
df = pd.concat(dataframes, ignore_index=True)
print("Total transactions:", len(df))

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------

df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

# Robust feature engineering with error handling
for col in ['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe. Check your data files.")

df['HOUR'] = df['TX_DATETIME'].dt.hour
df['DAY'] = df['TX_DATETIME'].dt.day
df['MONTH'] = df['TX_DATETIME'].dt.month

df['CUSTOMER_ENC'] = LabelEncoder().fit_transform(df['CUSTOMER_ID'].astype(str))
df['TERMINAL_ENC'] = LabelEncoder().fit_transform(df['TERMINAL_ID'].astype(str))

features = ['TX_AMOUNT', 'HOUR', 'DAY', 'MONTH', 'CUSTOMER_ENC', 'TERMINAL_ENC']
for feat in features:
    if feat not in df.columns:
        raise KeyError(f"Feature '{feat}' missing after engineering.")
X = df[features]
y = df['TX_FRAUD']

# -------------------------------
# STEP 3: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# STEP 4: Train Model
# -------------------------------

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# -------------------------------
# STEP 5: Evaluation
# -------------------------------

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Confusion Matrix Plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
