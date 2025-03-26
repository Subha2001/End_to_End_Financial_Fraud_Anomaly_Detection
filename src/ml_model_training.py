import pandas as pd
from xgboost import XGBClassifier
import joblib
import mlflow
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

def train_and_save_xgboost_model(X_train, y_train, X_val, y_val, model_path='model/XGBoost_model.joblib'):
    """
    Trains an XGBoost model, saves it, and logs evaluation metrics to MLflow.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        model_path (str): Path to save the trained model.
    """
    
    # Set tracking URI to lock Artifacts in MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set the MLflow experiment name
    mlflow.set_experiment("XGBoost_Model_Metrices")

    # Best hyperparameters obtained from tuning
    best_params = {
        "subsample": 0.7,
        "scale_pos_weight": 5,
        "n_estimators": 200,
        "min_child_weight": 4,
        "max_depth": 7,
        "learning_rate": 0.06,
        "gamma": 0,
        "colsample_bytree": 0.6,
    }

    # Create the XGBoost classifier using the best parameters
    xgb_clf = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        **best_params
    )

    # Fit the model on the training data (using the first 3,500,000 rows)
    xgb_clf.fit(X_train[:3500000], y_train[:3500000])

    # Save the trained model to the specified path
    joblib.dump(xgb_clf, model_path)
    print(f"XGBoost model trained and saved to: {model_path}")

    # Evaluate the model on the validation set
    y_pred = xgb_clf.predict(X_val)
    y_pred_proba = xgb_clf.predict_proba(X_val)[:, 1]

    # Calculate metrics
    cm = confusion_matrix(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_proba)

    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)  # log the parameters
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("mcc", mcc)
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_dict(cm.tolist(), "confusion_matrix")  # log confusion matrix
        mlflow.log_artifact(model_path)  # log the model

'''
# For Example
if __name__ == "__main__":
    # Load the CSV files into DataFrames/Series
    X_train_resampled = pd.read_csv(r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\artifacts\X_train_resampled.csv")
    y_train_resampled = pd.read_csv(r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\artifacts\y_train_resampled.csv")
    X_val_transformed = pd.read_csv(r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\artifacts\X_val_transformed.csv")
    y_val = pd.read_csv(r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\artifacts\y_val.csv")
    
    # If y_train and y_val are loaded as DataFrames with a single column,
    # you might want to convert them to Series:
    y_train_resampled = y_train_resampled.squeeze()
    y_val = y_val.squeeze()
    
    train_and_save_xgboost_model(
        X_train=X_train_resampled, 
        y_train=y_train_resampled, 
        X_val=X_val_transformed, 
        y_val=y_val
    )
    '''