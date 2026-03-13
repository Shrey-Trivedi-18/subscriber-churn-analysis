# ============================================
# Predictive Modeling (KNN)
# ============================================

# imports
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


def train_model():
    # --------------------------------------------
    # 1. Select features and target
    # --------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df_clean = df.dropna()

    X = df_clean[[
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "InternetService"
    ]]
    y = df_clean["Churn"]

    # --------------------------------------------
    # 2. Encode categorical features
    # --------------------------------------------
    X = pd.get_dummies(X, drop_first=True)

    # --------------------------------------------
    # 3. Train / test split
    # --------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------
    # 4. Scale features
    # --------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------
    # 5. Train KNN model
    # --------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # --------------------------------------------
    # 6. Make predictions
    # --------------------------------------------
    y_pred = knn.predict(X_test_scaled)
    y_prob = knn.predict_proba(X_test_scaled)[:, 1]

    return y_test, y_pred, y_prob


if __name__ == "__main__":
    y_test, y_pred, y_prob = train_model()

    # --------------------------------------------
    # 7. Evaluate model
    # --------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    # --------------------------------
    # 11. Confusion Matrix
    # --------------------------------
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"))
    plt.close()

    print("ROC AUC:", roc)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
