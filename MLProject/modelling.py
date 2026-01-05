import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    mlflow.set_experiment("creditcard-default")
    
    with mlflow.start_run():
    # =========================
    # Load dataset
    # =========================
    try:
        X_train = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_train.csv")
        X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
        y_train = pd.read_csv(
            "CreditCardDefaultDataset_preprocessing/y_train.csv"
        ).values.ravel()
        y_test = pd.read_csv(
            "CreditCardDefaultDataset_preprocessing/y_test.csv"
        ).values.ravel()
    except FileNotFoundError:
        print("❌ Dataset preprocessing tidak ditemukan.")
        return

    # =========================
    # Training
    # =========================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # =========================
    # Logging (AMAN)
    # =========================
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ Training selesai")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()
