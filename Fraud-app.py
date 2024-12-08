import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection Predictor", 
    page_icon="💳", 
    layout="wide"
)

# Constants (adjust these based on your model)
FRAUD_THRESHOLD = 0.30
RANDOM_STATE = 123

def load_model(model_path='best_fraud_model.joblib'):
    """
    Load the pre-trained model with robust error handling
    """
    # List all files in the current directory for debugging
    st.sidebar.info(f"Files in current directory: {os.listdir('.')}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file {model_path} not found!")
        st.sidebar.warning("Available files: " + ", ".join(os.listdir('.')))
        return None
    
    try:
        model = joblib.load(model_path)
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

def preprocess_data(df, predictors, categorical_cols, continuous_cols):
    """
    Preprocess input data using the same transformer used in training
    """
    # Create column transformer
    pre = make_column_transformer(
        (StandardScaler(), continuous_cols),
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        remainder="passthrough"
    )
    
    # Select only the predictors used during training
    X = df[predictors]
    return X

def predict_fraud(model, X):
    """
    Predict fraud probabilities and apply custom threshold
    """
    # Get probability scores
    y_proba = model.predict_proba(X)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_proba > FRAUD_THRESHOLD).astype(int)
    
    return y_proba, y_pred

def main():
    st.title("🕵️ Fraud Detection Predictor")
    
    # Sidebar for model and data configuration
    st.sidebar.header("Model Configuration")
    
    # TODO: REPLACE THESE WITH YOUR ACTUAL COLUMN CONFIGURATIONS
    predictors = [
        'transaction_amount', 'is_international', 
        'time_of_day', 'merchant_category'
    ]
    categorical_cols = ['is_international', 'merchant_category']
    continuous_cols = ['transaction_amount', 'time_of_day']
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File for Fraud Prediction", 
        type=['csv'],
        help="Upload a CSV file with the same columns used during model training"
    )
    
    # Model loading
    model = load_model()
    if model is None:
        st.sidebar.error("Could not load the model. Please check the model file.")
        return

    # Main prediction workflow
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            missing_cols = set(predictors) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return
            
            # Preprocess data
            X = preprocess_data(df, predictors, categorical_cols, continuous_cols)
            
            # Predict
            probabilities, predictions = predict_fraud(model, X)
            
            # Add results to dataframe
            df['fraud_probability'] = probabilities
            df['is_predicted_fraud'] = predictions
            
            # Display results
            st.subheader("Prediction Results")
            
            # Metrics
            total_records = len(df)
            predicted_fraud = sum(predictions)
            fraud_percentage = (predicted_fraud / total_records) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", total_records)
            col2.metric("Predicted Fraudulent", predicted_fraud)
            col3.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
            
            # Detailed results table
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Streamlit app entry point
if __name__ == '__main__':
    main()
