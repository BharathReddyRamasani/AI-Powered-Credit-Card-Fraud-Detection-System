import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix

def train_logistic_regression(X_train, y_train, class_weight='balanced', random_state=42):
    """
    Trains a Logistic Regression model adjusted for class imbalance.
    """
    print("Training Logistic Regression model...")
    # class_weight='balanced' automatically weights classes inversely proportional to class frequencies
    lr = LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=1000)
    lr.fit(X_train, y_train)
    print("Logistic Regression training complete.")
    return lr

def train_random_forest(X_train, y_train, class_weight='balanced_subsample', n_estimators=100, max_depth=10, random_state=42):
    """
    Trains a Random Forest classifier adjusted for class imbalance.
    Max depth is capped to keep the pickled model file size small for Git.
    """
    print(f"Training Random Forest model (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Random Forest training complete.")
    return rf

def evaluate_model(model, X_test, y_test, is_anomaly_detector=False):
    """
    Evaluates the model on test data and returns a dictionary of metrics.
    """
    if is_anomaly_detector:
        # Anomaly detectors predict -1 for anomalies (fraud) and 1 for normal
        # We need to map their predictions and scores to 1 (fraud) and 0 (normal)
        pass
        
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For model models that don't output probabilities, use decision function if available
        if hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            # Normalize to 0-1 range
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
        else:
            y_prob = y_pred
            
    # Calculate Precision, Recall, thresholds
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # F1 score
    f1 = 2 * (tp) / (2 * tp + fp + fn + 1e-9)
    prec_score = tp / (tp + fp + 1e-9)
    recall_score = tp / (tp + fn + 1e-9)
    
    metrics = {
        "precision_score": float(prec_score),
        "recall_score": float(recall_score),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_prob": y_prob.tolist() if isinstance(y_prob, np.ndarray) else list(y_prob)
    }
    
    return metrics
