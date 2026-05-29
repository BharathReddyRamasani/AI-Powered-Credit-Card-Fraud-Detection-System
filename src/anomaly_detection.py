import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix

class IsolationForestWrapper:
    """
    Wrapper for sklearn's IsolationForest to align its API with standard binary classifiers.
    """
    def __init__(self, contamination=0.0017, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def fit(self, X_train, y_train=None):
        """
        Fits the Isolation Forest. If y_train is provided, we train ONLY on the normal transactions (y_train == 0),
        which is standard practice for unsupervised anomaly detection.
        """
        print("Training Isolation Forest anomaly detection model...")
        if y_train is not None:
            # Train only on normal transactions
            X_normal = X_train[y_train == 0]
            print(f"Fitting Isolation Forest on {X_normal.shape[0]} normal transactions...")
            self.model.fit(X_normal)
        else:
            print(f"Fitting Isolation Forest on all {X_train.shape[0]} transactions...")
            self.model.fit(X_train)
        print("Isolation Forest training complete.")
        return self
        
    def predict(self, X):
        """
        Predicts binary labels: 1 for fraud (anomaly), 0 for normal.
        sklearn outputs -1 for anomalies and 1 for inliers.
        """
        preds = self.model.predict(X)
        # Map: 1 (inlier) -> 0 (normal), -1 (outlier) -> 1 (fraud)
        binary_preds = np.where(preds == -1, 1, 0)
        return binary_preds
        
    def predict_proba(self, X):
        """
        Estimates 'anomaly probability' or confidence score.
        IsolationForest does not have a native predict_proba.
        We can use the decision_function score.
        decision_function returns negative values for anomalies and positive for inliers.
        We transform it so that a higher score means higher probability of being fraud.
        """
        decision_scores = self.model.decision_function(X)
        # decision_function score is: higher is normal, lower is anomalous.
        # We invert it: anomaly_score = -decision_score
        anomaly_scores = -decision_scores
        
        # Min-Max normalize anomaly scores to [0, 1] range to resemble probabilities
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        if max_score - min_score > 0:
            probs = (anomaly_scores - min_score) / (max_score - min_score)
        else:
            probs = np.zeros_like(anomaly_scores)
            
        # Reshape to match predict_proba output shape (n_samples, 2)
        # class 0 prob, class 1 prob
        proba_matrix = np.column_stack((1 - probs, probs))
        return proba_matrix

def train_isolation_forest(X_train, y_train, contamination=0.0017, random_state=42):
    """
    Trains an Isolation Forest model using the wrapper.
    """
    wrapper = IsolationForestWrapper(contamination=contamination, random_state=random_state)
    wrapper.fit(X_train, y_train)
    return wrapper

def evaluate_anomaly_detector(model, X_test, y_test):
    """
    Evaluates the Isolation Forest model on test data and returns metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Anomaly score probability
    
    # Calculate PR curve and AUC
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
