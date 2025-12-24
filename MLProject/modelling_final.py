import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

dagshub.init(
    repo_owner="Geversonvx",
    repo_name="Eksperimen_SML_GilangPutraFirmansyah",
    mlflow=True,
)
# =============================
# Load data preprocessing
# =============================
X_train = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_train.csv")
X_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/X_test.csv")
y_train = pd.read_csv(
    "CreditCardDefaultDataset_preprocessing/y_train.csv"
).values.ravel()
y_test = pd.read_csv("CreditCardDefaultDataset_preprocessing/y_test.csv").values.ravel()

# =============================
# Model terbaik (hasil tuning)
# =============================
best_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
)
mlflow.set_experiment("CreditCard_Default_RF_DagsHub")

with mlflow.start_run(run_name="RandomForest_Final_Model"):

    # PARAMETER
    n_estimators = 200
    max_depth = 10

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )

    model.fit(X_train, y_train)

    # PREDIKSI
    y_pred = model.predict(X_test)

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # SIMPAN MODEL LOKAL
    os.makedirs("artifacts_extra", exist_ok=True)
    joblib.dump(model, "artifacts_extra/random_forest_final_model.joblib")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CreditCardDefault_RF_Model",
    )

    # ===== ARTEFAK TAMBAHAN (MINIMAL 2) =====

    # 1️⃣ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_path = "artifacts_extra/confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=False)
    mlflow.log_artifact(cm_path)

    # 2️⃣ Model Final (.joblib)
    mlflow.log_artifact("artifacts_extra/random_forest_final_model.joblib")

    print("Training & logging ke DagsHub selesai")
