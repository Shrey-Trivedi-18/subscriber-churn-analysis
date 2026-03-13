import os
import sys

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from models.model import y_test, y_pred


sns.set(style="whitegrid")

# --------------------------------
# Resolve project paths
# --------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")

# --------------------------------
# Load dataset
# --------------------------------
df = pd.read_csv(DATA_PATH)
df["Churn"] = df["Churn"].astype(str)
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# --------------------------------
# 1. Churn Distribution
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig(os.path.join(IMAGE_DIR, "churn_distribution.png"))
plt.close()

# --------------------------------
# 2. Churn by Gender
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Churn by Gender")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_gender.png"))
plt.close()

# --------------------------------
# 3. Churn by Internet Service
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_internet.png"))
plt.close()

# --------------------------------
# 4. Churn by Senior Citizen
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="SeniorCitizen", hue="Churn", data=df)
plt.title("Churn by Senior Citizen")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_senior.png"))
plt.close()

# --------------------------------
# 5. Churn by Dependents
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Dependents", hue="Churn", data=df)
plt.title("Churn by Dependents")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_dependents.png"))
plt.close()

# --------------------------------
# 6. Churn by Online Security
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="OnlineSecurity", hue="Churn", data=df)
plt.title("Churn by Online Security")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_security.png"))
plt.close()

# --------------------------------
# 7. Churn by Tech Support
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="TechSupport", hue="Churn", data=df)
plt.title("Churn by Tech Support")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_techsupport.png"))
plt.close()

# --------------------------------
# 8. Churn by Paperless Billing
# --------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="PaperlessBilling", hue="Churn", data=df)
plt.title("Churn by Paperless Billing")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_billing.png"))
plt.close()

# --------------------------------
# 9. Monthly Charges vs Churn
# --------------------------------

plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig(os.path.join(IMAGE_DIR, "monthly_charges.png"))
plt.close()

# --------------------------------
# 10. Tenure Distribution by Churn
# --------------------------------

plt.figure(figsize=(6,4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30)
plt.title("Tenure Distribution by Churn")
plt.savefig(os.path.join(IMAGE_DIR, "tenure_churn.png"))
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(x="risk_segment", hue="Churn", data=df)
plt.title("Churn by Risk Segment")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_risk_segment.png"))
plt.close()

# --------------------------------
# 11. Confusion Matrix
# --------------------------------

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"))
plt.close()
# Load dataset

# --------------------------------
# 12. Churn by Contract
# --------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_contract.png"))
plt.close()

# --------------------------------
# 13. Churn by Payment Method
# --------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.savefig(os.path.join(IMAGE_DIR, "churn_by_payment.png"))
plt.close()

print("All graphs saved to:", IMAGE_DIR)

