import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def predict_fraud(input_data, model_path='model/XGBoost_model.joblib'):
    """
    Loads a saved XGBoost model and predicts fraud based on input data.
    """
    try:
        # Load the trained XGBoost model
        loaded_model = joblib.load(model_path)
        
        # OneHot encode the 'type' with a fixed set of categories.
        encoder = OneHotEncoder(
            categories=[['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']],
            sparse_output=False,
            handle_unknown='ignore'
        )
        # Use transform on the array (the encoder will know to output 5 columns)
        type_encoded = encoder.fit_transform(np.array(input_data['type']).reshape(1, 1))
        
        # Create DataFrame from input data, with the correct column order.
        input_df = pd.DataFrame(
            [[
                input_data['amount'],
                input_data['oldbalanceOrg'],
                input_data['newbalanceOrg'],
                input_data['oldbalanceDest'],
                input_data['newbalanceDest'],
                input_data['balanceDeltaOrg'],
                input_data['balanceDeltaDest'],
            ]],
            columns=[
                'num__amount',
                'num__oldbalanceOrg',
                'num__newbalanceOrig',
                'num__oldbalanceDest',
                'num__newbalanceDest',
                'num__balanceDeltaOrg',
                'num__balanceDeltaDest',
            ],
        )

        # Define the type columns as expected.
        type_columns = [
            'cat__type_CASH_IN',
            'cat__type_CASH_OUT',
            'cat__type_DEBIT',
            'cat__type_PAYMENT',
            'cat__type_TRANSFER',
        ]
        type_df = pd.DataFrame(type_encoded, columns=type_columns)

        # Concatenate the DataFrames
        input_df = pd.concat([input_df, type_df], axis=1)
        
        # Reorder the columns to the correct order.
        input_df = input_df[['num__amount','num__oldbalanceOrg','num__newbalanceOrig',
                             'num__oldbalanceDest','num__newbalanceDest','num__balanceDeltaOrg',
                             'num__balanceDeltaDest','cat__type_CASH_IN','cat__type_CASH_OUT',
                             'cat__type_DEBIT','cat__type_PAYMENT','cat__type_TRANSFER']]
        
        # Rename the columns to match the model's expected order.
        input_df.columns = ['0','1','2','3','4','5','6','7','8','9','10','11']
        
        # Predict the probability of fraud
        fraud_probability = loaded_model.predict_proba(input_df)[:, 1]
        
        # Convert probability to binary prediction (1 for fraud, 0 for not fraud)
        fraud_prediction = (fraud_probability >= 0.5).astype(int)[0]
        
        return fraud_prediction

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

'''
# For Example
if __name__ == "__main__":
    # User input
    type_input = input("Enter the Type: ")
    amount_input = float(input("Enter the Amount: "))
    oldbalanceOrg_input = float(input("Enter the Old Balance Org: "))
    newbalanceOrg_input = float(input("Enter the New Balance Org: "))
    oldbalanceDest_input = float(input("Enter the Old Balance Dest: "))
    newbalanceDest_input = float(input("Enter the New Balance Dest: "))
    
    balanceDeltaOrg_input = oldbalanceOrg_input - newbalanceOrg_input
    balanceDeltaDest_input = oldbalanceDest_input - newbalanceDest_input
    
    # Prepare input data
    input_data = {
        'type': type_input,
        'amount': amount_input,
        'oldbalanceOrg': oldbalanceOrg_input,
        'newbalanceOrg': newbalanceOrg_input,
        'oldbalanceDest': oldbalanceDest_input,
        'newbalanceDest': newbalanceDest_input,
        'balanceDeltaOrg': balanceDeltaOrg_input,
        'balanceDeltaDest': balanceDeltaDest_input,
    }
    
    # Make prediction
    prediction = predict_fraud(input_data)
    
    if prediction is not None:
        label_map = {0: 'No', 1: 'Yes'}
        print(f"Prediction: {label_map[prediction]}")
        '''