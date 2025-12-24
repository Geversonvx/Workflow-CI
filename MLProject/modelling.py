import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Init DagsHub + MLflow
# =========================
dagshub.init(
    repo_owner="Geversonvx",
    repo_name="Eksperimen_SML_GilangPutraFirmansyah",
    mlflow=True,
)


def main():
    # =========================
    # Load data preprocessing
    # =========================
    X_train = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_train.csv")
    X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
    y_train = pd.read_csv(
        "CreditCardDefaultDataset_preprocessing/y_train.csv"
    ).values.ravel()
    y_test = pd.read_csv(
        "CreditCardDefaultDataset_preprocessing/y_test.csv"
    ).values.ravel()

    # =========================
    # MLflow experiment
    # =========================
    mlflow.set_experiment("Credit_Card_Default_Basic_Model")

    with mlflow.start_run():
        # =========================
        # Training
        # =========================
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)

        # =========================
        # Evaluation
        # =========================
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # =========================
        # Manual logging (WAJIB)
        # =========================
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # =========================
        # Log & REGISTER model
        # =========================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditCardDefault_RF_Model",
        )

        print("Model trained & registered successfully")


if __name__ == "__main__":
    main()
