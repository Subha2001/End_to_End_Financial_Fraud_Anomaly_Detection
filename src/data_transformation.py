import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import data_ingestion  # Import your data_ingestion module
import os

def create_new_features(df):
    """Creates new features based on balance differences."""
    df['balanceDeltaOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDeltaDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    return df

def drop_irrelevant_features(df):
    """Drops irrelevant features from the DataFrame."""
    df = df.drop(['step', 'nameOrig', 'nameDest'], axis=1)
    return df

def perform_train_val_test_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """Splits the data into training, validation, and testing sets."""
    # Step 1: Split into train+validation and test sets
    X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = df['isFraud']
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Step 2: Split train+validation into train and validation sets
    val_ratio = val_size / (1 - test_size)  # Adjust validation size relative to train+validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test):
    """Applies StandardScaler and OneHotEncoder to the training, validation, and testing data."""
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balanceDeltaOrg', 'balanceDeltaDest']
    categorical_features = ['type']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    return X_train_transformed, X_val_transformed, X_test_transformed

def apply_smote(X_train, y_train, sampling_strategy=0.4, k_neighbors=3, random_state=42):
    """Applies SMOTE to the training data to handle class imbalance."""
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def save_data_to_csv(X_train_resampled, y_train_resampled, X_val_transformed, y_val, X_test_transformed, y_test, artifact_dir="artifacts"):
    """Saves the preprocessed and resampled data to CSV files in the specified directory."""

    # Create the directory if it doesn't exist
    os.makedirs(artifact_dir, exist_ok=True)

    # Convert to DataFrames for saving to CSV
    X_train_resampled_df = pd.DataFrame(X_train_resampled)
    X_val_transformed_df = pd.DataFrame(X_val_transformed)
    X_test_transformed_df = pd.DataFrame(X_test_transformed)
    y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=['isFraud'])
    y_val_df = pd.DataFrame(y_val, columns=['isFraud'])
    y_test_df = pd.DataFrame(y_test, columns=['isFraud'])

    # Save to CSV files
    X_train_resampled_df.to_csv(os.path.join(artifact_dir, "X_train_resampled.csv"), index=False)
    X_val_transformed_df.to_csv(os.path.join(artifact_dir, "X_val_transformed.csv"), index=False)
    X_test_transformed_df.to_csv(os.path.join(artifact_dir, "X_test_transformed.csv"), index=False)
    y_train_resampled_df.to_csv(os.path.join(artifact_dir, "y_train_resampled.csv"), index=False)
    y_val_df.to_csv(os.path.join(artifact_dir, "y_val.csv"), index=False)
    y_test_df.to_csv(os.path.join(artifact_dir, "y_test.csv"), index=False)

    print(f"Processed data saved to CSV files in '{artifact_dir}' folder.")

def process_data_pipeline(df):
    """Executes the data processing pipeline."""

    # 1. Create new features
    df = create_new_features(df.copy())

    # 2. Drop irrelevant features
    df = drop_irrelevant_features(df.copy())

    # 3. Train-validation-test split
    X_train, X_val, X_test, y_train, y_val, y_test = perform_train_val_test_split(df)

    # 4. Preprocess data
    X_train_transformed, X_val_transformed, X_test_transformed = preprocess_data(X_train, X_val, X_test)

    # 5. Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train_transformed, y_train)

    # 6. Save data to CSV
    save_data_to_csv(X_train_resampled, y_train_resampled, X_val_transformed, y_val, X_test_transformed, y_test)

    return X_train_resampled, y_train_resampled, X_val_transformed, y_val, X_test_transformed, y_test

'''
# For Example
if __name__ == "__main__":
    # Load data using the function from data_ingestion.py
    df = data_ingestion.load_financial_fraud_data()

    if df is not None:
        X_train_resampled, y_train_resampled, X_val_transformed, y_val, X_test_transformed, y_test = process_data_pipeline(df)

        print("X_train_resampled shape:", X_train_resampled.shape)
        print("y_train_resampled shape:", y_train_resampled.shape)
        print("X_val_transformed shape:", X_val_transformed.shape)
        print("y_val shape:", y_val.shape)
        print("X_test_transformed shape:", X_test_transformed.shape)
        print("y_test shape:", y_test.shape)
    else:
        print("Failed to load data from data_ingestion.py")
        '''