import joblib
import mlflow
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

def evaluate_xgboost_model(X_test, y_test, 
                           model_path = r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\model\XGBoost_model.joblib"):
    """
    Loads a saved XGBoost model, evaluates it on test data, and logs metrics to MLflow.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_path (str): Path to the saved XGBoost model.
    """
    # Set tracking URI to lock Artifacts in MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set the MLflow experiment name
    mlflow.set_experiment("XGBoost_Evaluation_Metrices")

    try:
        # Load the trained XGBoost model
        loaded_model = joblib.load(model_path)

        # Predict on the test data
        y_pred = loaded_model.predict(X_test)
        y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Log metrics to MLflow
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("mcc", mcc)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_dict(cm.tolist(), "confusion_matrix")  # Log confusion matrix
            mlflow.log_artifact(model_path) # log the model
            print("Evaluation metrics logged to MLflow.")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

'''
# For Example
if __name__ == "__main__":
    import pandas as pd

    # Load the CSV files into DataFrames/Series
    X_test_transformed = pd.read_csv(r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\artifacts\X_test_transformed.csv")
    y_test = pd.read_csv(r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\artifacts\y_test.csv")

    evaluate_xgboost_model(X_test_transformed, y_test)
    '''