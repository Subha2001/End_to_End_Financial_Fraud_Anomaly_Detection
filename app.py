import streamlit as st
from src.prediction import predict_fraud

# Set the page configuration
st.set_page_config(page_title="Financial Fraud Detection", layout="centered")

# Inject custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main background */
    .reportview-container {
        background-color: #F0F2F6;
    }
    /* Title style */
    .title {
        font-family: 'Arial', sans-serif;
        color: #00FFFF;
        font-size: 36px;
        text-align: center;
        margin-bottom: 0px;
    }
    /* Subtitle style */
    .subtitle {
        font-family: 'Arial', sans-serif;
        color: #FF0000;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Display the title and subtitle
st.markdown('<div class="title">Financial Fraud Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the transaction details below to get the prediction.</div>', unsafe_allow_html=True)

# Expandable instructions section
with st.expander("How to Use This App"):
    st.write("""
        - **Transaction Type:** Select from CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER.
        - **Amount:** Enter the transaction amount.
        - **Old/New Balance Org/Dest:** Fill in the account balances.
        - The app automatically computes the balance deltas.
        - Click **Predict Fraud** to see the outcome.
    """)

st.write("")  # Spacer

# Layout inputs in two columns for a cleaner look
col1, col2 = st.columns(2)
with col1:
    transaction_type = st.selectbox("Transaction Type", 
                                      ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Amount", value=0.0, min_value=0.0, step=0.01)
    oldbalanceOrg = st.number_input("Old Balance Org [Sender’s balance before transaction]", value=0.0, min_value=0.0, step=0.01)

with col2:
    newbalanceOrg = st.number_input("New Balance Org [Sender’s balance after transaction]", value=0.0, min_value=0.0, step=0.01)
    oldbalanceDest = st.number_input("Old Balance Dest [Receiver’s balance after transaction]", value=0.0, min_value=0.0, step=0.01)
    newbalanceDest = st.number_input("New Balance Dest [Receiver’s balance after transaction]", value=0.0, min_value=0.0, step=0.01)

# Calculate balance deltas
balanceDeltaOrg = oldbalanceOrg - newbalanceOrg
balanceDeltaDest = oldbalanceDest - newbalanceDest

st.write("**Balance Delta Org**: ", balanceDeltaOrg)
st.write("**Balance Delta Dest**: ", balanceDeltaDest)

st.write("")  # Spacer

# Prediction button and output area
if st.button("Predict Fraud"):
    with st.spinner("Analyzing transaction..."):
        # Assemble the input dictionary
        input_data = {
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrg': newbalanceOrg,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'balanceDeltaOrg': balanceDeltaOrg,
            'balanceDeltaDest': balanceDeltaDest,
        }
        # Get the prediction from the model
        prediction = predict_fraud(input_data)

    st.write("")  # Spacer after spinner

    # Display prediction with custom styling
    if prediction is None:
        st.error("An error occurred while predicting. Check the logs.")
    else:
        if prediction == 1:
            st.markdown(
                '<p style="background-color: #FF0000; color: white; font-family: \'Arial\', sans-serif; '
                'font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 8px;">'
                'Prediction: Fraud Detected</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p style="background-color: #00AA00; color: white; font-family: \'Arial\', sans-serif; '
                'font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 8px;">'
                'Prediction: No Fraud Detected</p>',
                unsafe_allow_html=True
            )