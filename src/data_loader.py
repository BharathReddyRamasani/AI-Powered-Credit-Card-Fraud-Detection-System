import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def download_and_load_data(save_dir="data"):
    """
    Downloads the credit card fraud dataset from Kaggle programmatically
    and loads it into a pandas DataFrame.
    """
    import kagglehub
    print("Downloading credit card fraud dataset using kagglehub...")
    try:
        # Download latest version of mlg-ulb/creditcardfraud
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"Dataset downloaded to path: {path}")
        
        # Locate the CSV file in the downloaded path
        files = os.listdir(path)
        csv_file = [f for f in files if f.endswith('.csv')][0]
        csv_path = os.path.join(path, csv_file)
        
        print(f"Loading dataset from: {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded dataset with shape: {df.shape}")
        
        # Optionally create local copy for offline use/fast access
        os.makedirs(save_dir, exist_ok=True)
        local_csv = os.path.join(save_dir, "creditcard.csv")
        if not os.path.exists(local_csv):
            # Save a compressed version or sample if too large, but we can copy the full file
            print(f"Creating local copy of CSV at {local_csv}...")
            df.to_csv(local_csv, index=False)
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try to load local file if download fails
        local_csv = os.path.join(save_dir, "creditcard.csv")
        if os.path.exists(local_csv):
            print(f"Loading cached local copy from {local_csv}...")
            return pd.read_csv(local_csv)
        raise e

class FraudDataPreprocessor:
    """
    Preprocesses the credit card dataset: scales 'Time' and 'Amount' and splits data.
    """
    def __init__(self):
        self.time_scaler = StandardScaler()
        self.amount_scaler = StandardScaler()
        
    def fit_transform(self, df):
        """
        Fits scalers on Time and Amount and transforms them.
        """
        processed_df = df.copy()
        
        # Scale 'Time' and 'Amount' columns
        processed_df['scaled_time'] = self.time_scaler.fit_transform(processed_df[['Time']])
        processed_df['scaled_amount'] = self.amount_scaler.fit_transform(processed_df[['Amount']])
        
        # Drop raw 'Time' and 'Amount'
        processed_df = processed_df.drop(['Time', 'Amount'], axis=1)
        
        # Reorder columns so scaled columns are at the front (or just return)
        # Shift target 'Class' to the end
        cols = [c for c in processed_df.columns if c != 'Class'] + ['Class']
        processed_df = processed_df[cols]
        
        return processed_df
        
    def transform(self, df):
        """
        Transforms Time and Amount using fitted scalers (for inference).
        """
        processed_df = df.copy()
        if 'Time' in processed_df.columns:
            processed_df['scaled_time'] = self.time_scaler.transform(processed_df[['Time']])
            processed_df = processed_df.drop(['Time'], axis=1)
        if 'Amount' in processed_df.columns:
            processed_df['scaled_amount'] = self.amount_scaler.transform(processed_df[['Amount']])
            processed_df = processed_df.drop(['Amount'], axis=1)
            
        # Reorder if Class is present
        if 'Class' in processed_df.columns:
            cols = [c for c in processed_df.columns if c != 'Class'] + ['Class']
            processed_df = processed_df[cols]
            
        return processed_df

def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """
    Preprocesses the dataframe and returns stratified training and test splits.
    """
    preprocessor = FraudDataPreprocessor()
    processed_df = preprocessor.fit_transform(df)
    
    # Split into features and target
    X = processed_df.drop('Class', axis=1)
    y = processed_df['Class']
    
    # Stratified split to maintain class balance in both training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor
