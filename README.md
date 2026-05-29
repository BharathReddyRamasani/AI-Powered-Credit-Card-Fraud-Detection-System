# 💳 AI-Powered Credit Card Fraud Detection System

An interactive, high-fidelity machine learning dashboard designed to demonstrate the detection of fraudulent credit card transactions. This project implements and compares **Logistic Regression** (supervised classifier with class balancing), **Random Forest** (supervised tree ensemble), and **Isolation Forest** (unsupervised anomaly detector), alongside **Principal Component Analysis (PCA)** visualization.

This repository is optimized for educational demonstration and is fully ready to be deployed to **Streamlit Community Cloud**.

---

## 🌟 Features

* **📊 Live Metrics Dashboard**: Real-time stats on dataset volume, baseline fraud proportions (0.17%), and transaction amounts.
* **🌌 Interactive PCA Visualizer**: Interactive 2D/3D scatter plotting tool to visualize class separation between legitimate and fraudulent transactions.
* **📈 Side-by-Side Model Evaluation**: Compare models using standard metrics including **Precision**, **Recall**, **F1-Score**, and **AUPRC (Area Under Precision-Recall Curve)**.
* **🧬 Confusion Matrix Heatmaps**: Beautiful heatmaps detailing True Positives, False Positives, True Negatives, and False Negatives.
* **⚡ Transaction Sandbox (Live Predictor)**: Test custom parameters (Amount, Time, and PCA features) manually or load pre-built scenario presets (e.g., *Normal Legitimate*, *Suspicious Fraud*, *High Amount Anomaly*).

---

## 🛠️ Tech Stack & Setup

### Requirements

Install dependencies listed in [requirements.txt](requirements.txt):
* `python >= 3.8`
* `pandas`
* `numpy`
* `scikit-learn`
* `streamlit`
* `plotly`
* `kagglehub`
* `joblib`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BharathReddyRamasani/AI-Powered-Credit-Card-Fraud-Detection-System.git
   cd AI-Powered-Credit-Card-Fraud-Detection-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training pipeline**:
   This script will programmatically download the Kaggle dataset (`mlg-ulb/creditcardfraud`) using `kagglehub`, fit the preprocessing scalers, train all 3 models, and serialize the artifacts:
   ```bash
   python train_pipeline.py
   ```

4. **Launch the Streamlit web application**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Model Architecture & Takeaways

### 1. Logistic Regression
* **Approach**: Supervised linear classification.
* **Class Imbalance Strategy**: Trained using `class_weight='balanced'` which penalizes errors in the minority class (fraud) heavily.
* **Performance**: **91.84% Recall** and **6.09% Precision**.
* **Key Takeaway**: By shifting weights, it catches almost all frauds (high Recall) but suffers from high false alarms (low Precision).

### 2. Random Forest Classifier
* **Approach**: Supervised tree ensemble.
* **Class Imbalance Strategy**: Trained with `class_weight='balanced_subsample'` and restricted complexity (`max_depth=10`, `n_estimators=50`) to keep model storage clean and prevent overfitting.
* **Performance**: **81.63% Recall** and **79.21% Precision** (AUPRC = **0.8251**).
* **Key Takeaway**: Excellent balanced metric, capturing a high proportion of fraud with minimal customer annoyance (false positives).

### 3. Isolation Forest
* **Approach**: Unsupervised Anomaly Detection.
* **Class Imbalance Strategy**: Trained **strictly on normal transactions** to model inlier distribution, mapping outliers as anomalies.
* **Performance**: **24.49% Recall** and **18.60% Precision**.
* **Key Takeaway**: Since it doesn't use the 'Class' label during training, its scores are lower. However, it is **indispensable** for identifying zero-day/novel fraud signatures that supervised models would miss.

---

## 📂 Project Structure

```text
AI-Powered-Credit-Card-Fraud-Detection-System/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Kaggle dataset download, scaling, stratified splitting
│   ├── models.py             # Supervised models training (LR & RF)
│   ├── anomaly_detection.py  # Unsupervised Isolation Forest wrapper
│   └── utils.py              # Serialization helpers and Plotly curve drawing
├── models/                   # Serialized pipeline weights (.pkl) and JSON statistics
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   ├── metrics.json
│   └── ui_plot_data.csv      # Representative sub-sample for UI plot speeds
├── app.py                    # Web Application interface
├── train_pipeline.py         # End-to-end ML CLI script
├── requirements.txt          # Python packages list
├── .gitignore                # System and Cache Git exclusions
└── README.md                 # Project documentation
```

---

## 🚀 Streamlit Cloud Deployment

This repository is optimized to deploy with one click on **Streamlit Community Cloud**:
1. Push all code to your GitHub repository:
   ```bash
   git add .
   git commit -m "feat: complete credit card fraud detection system with UI and ML models"
   git push origin main
   ```
2. Log in to [Streamlit Share](https://share.streamlit.io/).
3. Click **New App**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!** The cloud server will install dependencies from `requirements.txt`, run the dashboard, and programmatically load/cache the Kaggle dataset.