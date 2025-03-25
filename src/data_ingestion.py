import pandas as pd
import os

def load_financial_fraud_data(filepath = r"C:\End_to_End_Financial_Fraud_Anomaly_Detection\data\Financial_Fraud_Dataset.csv"):
    """
    Loads financial fraud data from a CSV file.

    Args:
        filepath (str): The path to the CSV file. Defaults to the specified path.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame, or None if an error occurs.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at: {filepath}")
        df = pd.read_csv(filepath)
        print("File loaded successfully!")  # Debugging output
        return df
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


# Example Usage
if __name__ == "__main__":
    df = load_financial_fraud_data()
    if df is not None:
        print("Data loaded successfully!")
        print(df.head()) # Print the first few rows of the data.

    else:
        print("Data loading failed.")