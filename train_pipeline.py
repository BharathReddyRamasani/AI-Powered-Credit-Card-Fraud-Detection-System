import os
import json
import pandas as pd
import numpy as np
from src.data_loader import download_and_load_data, prepare_train_test_data
from src.models import train_logistic_regression, train_random_forest, evaluate_model
from src.anomaly_detection import train_isolation_forest, evaluate_anomaly_detector
from src.utils import save_artifact

def main():
    # 1. Download and load data
    print("=== Step 1: Loading Data ===")
    df = download_and_load_data()
    
    # 2. Preprocess and split data
    print("\n=== Step 2: Preprocessing and Splitting ===")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training Fraud cases: {np.sum(y_train)} ({(np.sum(y_train)/len(y_train))*100:.4f}%)")
    print(f"Test Fraud cases: {np.sum(y_test)} ({(np.sum(y_test)/len(y_test))*100:.4f}%)")
    
    # Save the preprocessor
    save_artifact(preprocessor, "preprocessor.pkl")
    
    # 3. Train Models
    print("\n=== Step 3: Training Models ===")
    
    # 3.1 Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    save_artifact(lr_model, "logistic_regression.pkl")
    
    # 3.2 Random Forest (limit complexity to prevent giant git files)
    rf_model = train_random_forest(X_train, y_train, n_estimators=50, max_depth=10)
    save_artifact(rf_model, "random_forest.pkl")
    
    # 3.3 Isolation Forest (contamination = fraud rate in training set)
    fraud_rate = float(np.sum(y_train) / len(y_train))
    if_model = train_isolation_forest(X_train, y_train, contamination=fraud_rate)
    save_artifact(if_model, "isolation_forest.pkl")
    
    # 4. Evaluate Models
    print("\n=== Step 4: Evaluating Models ===")
    
    print("\nEvaluating Logistic Regression...")
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print(f"Logistic Regression - Precision: {lr_metrics['precision_score']:.4f}, Recall: {lr_metrics['recall_score']:.4f}, F1: {lr_metrics['f1_score']:.4f}, AUPRC: {lr_metrics['pr_auc']:.4f}")
    
    print("\nEvaluating Random Forest...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest - Precision: {rf_metrics['precision_score']:.4f}, Recall: {rf_metrics['recall_score']:.4f}, F1: {rf_metrics['f1_score']:.4f}, AUPRC: {rf_metrics['pr_auc']:.4f}")
    
    print("\nEvaluating Isolation Forest...")
    if_metrics = evaluate_anomaly_detector(if_model, X_test, y_test)
    print(f"Isolation Forest - Precision: {if_metrics['precision_score']:.4f}, Recall: {if_metrics['recall_score']:.4f}, F1: {if_metrics['f1_score']:.4f}, AUPRC: {if_metrics['pr_auc']:.4f}")
    
    # Save test labels and prediction probabilities/scores for visualization in Streamlit
    # This avoids loading huge CSVs and computing things dynamically in Streamlit.
    y_test_list = y_test.tolist()
    
    summary_metrics = {
        "dataset_stats": {
            "total_transactions": len(df),
            "total_fraud": int(df['Class'].sum()),
            "fraud_percentage": float(df['Class'].mean() * 100)
        },
        "logistic_regression": {
            "precision": lr_metrics["precision_score"],
            "recall": lr_metrics["recall_score"],
            "f1": lr_metrics["f1_score"],
            "pr_auc": lr_metrics["pr_auc"],
            "roc_auc": lr_metrics["roc_auc"],
            "confusion_matrix": lr_metrics["confusion_matrix"]
        },
        "random_forest": {
            "precision": rf_metrics["precision_score"],
            "recall": rf_metrics["recall_score"],
            "f1": rf_metrics["f1_score"],
            "pr_auc": rf_metrics["pr_auc"],
            "roc_auc": rf_metrics["roc_auc"],
            "confusion_matrix": rf_metrics["confusion_matrix"]
        },
        "isolation_forest": {
            "precision": if_metrics["precision_score"],
            "recall": if_metrics["recall_score"],
            "f1": if_metrics["f1_score"],
            "pr_auc": if_metrics["pr_auc"],
            "roc_auc": if_metrics["roc_auc"],
            "confusion_matrix": if_metrics["confusion_matrix"]
        },
        "y_test_sample": y_test_list[:10000],  # Save a sample of test targets and predictions to draw curves
        "y_probs_sample": {
            "logistic_regression": lr_metrics["y_prob"][:10000],
            "random_forest": rf_metrics["y_prob"][:10000],
            "isolation_forest": if_metrics["y_prob"][:10000]
        }
    }
    
    # Save a small subset of the raw data for visual representation (e.g. 5000 normal + all 492 fraud)
    print("\nSaving a representative subset of data for fast UI loading...")
    fraud_subset = df[df['Class'] == 1]
    normal_subset = df[df['Class'] == 0].sample(n=5000, random_state=42)
    ui_df = pd.concat([normal_subset, fraud_subset])
    os.makedirs("models", exist_ok=True)
    ui_df.to_csv("models/ui_plot_data.csv", index=False)
    
    # Save metrics JSON
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
