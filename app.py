import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Config
st.set_page_config(
    page_title="AI-Powered Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    
    .main-title-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-title-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    
    .card-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #F1F5F9;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        color: #475569;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        border-top: 3px solid #4F46E5 !important;
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }
    
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fraud {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .badge-legit {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions to load data and artifacts safely
@st.cache_resource
def load_models_and_scaler_v3():
    """Loads scalers and classifiers."""
    try:
        preprocessor = joblib.load("models/preprocessor.pkl")
        lr_model = joblib.load("models/logistic_regression.pkl")
        rf_model = joblib.load("models/random_forest.pkl")
        if_model = joblib.load("models/isolation_forest.pkl")
        return preprocessor, lr_model, rf_model, if_model
    except Exception as e:
        st.error(f"Error loading model binaries. Please make sure train_pipeline.py has run successfully. Details: {e}")
        return None, None, None, None

@st.cache_data
def load_metrics_and_plot_data_v3():
    """Loads metrics JSON and visual plotting subset."""
    metrics = None
    plot_df = None
    
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
            
    if os.path.exists("models/ui_plot_data.csv"):
        plot_df = pd.read_csv("models/ui_plot_data.csv")
        
    return metrics, plot_df

# Load all artifacts
preprocessor, lr_model, rf_model, if_model = load_models_and_scaler_v3()
metrics_data, plot_df = load_metrics_and_plot_data_v3()

# Render Header Title Banner
st.markdown("""
    <div class="main-title-container">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">💳 AI-Powered Credit Card Fraud Detection</h1>
        <p style="color: #E0E7FF; margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 300;">
            Interactive Machine Learning Dashboard using Logistic Regression, Random Forest, Isolation Forest, and PCA Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

if not preprocessor or not metrics_data or plot_df is None:
    st.warning("⚠️ Warning: Pre-trained artifacts not found or incomplete. Run the training pipeline first to initialize.")
    if st.button("🚀 Run Training Pipeline Now"):
        with st.spinner("Downloading dataset and training models... this may take 1-2 minutes."):
            import subprocess
            result = subprocess.run(["python", "train_pipeline.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Training pipeline ran successfully! Reloading page...")
                st.rerun()
            else:
                st.error(f"Pipeline error:\n{result.stderr}")
    st.stop()

# Sidebar Info Panel
st.sidebar.markdown("""
    <div class="card-container" style="background-color: #F8FAFC;">
        <h3 style="margin-top: 0;">ℹ️ Project Information</h3>
        <p style="font-size: 0.9rem; color: #475569;">
            This system demonstrates real-world banking AI techniques for fraud detection. It uses Kaggle's <b>creditcardfraud</b> dataset (284,807 transactions).
        </p>
        <p style="font-size: 0.9rem; color: #475569;">
            <b>Key Concepts Taught:</b>
            <ul>
                <li>Handling Highly Imbalanced Data</li>
                <li>Anomaly Detection (Unsupervised)</li>
                <li>Precision/Recall Trade-offs</li>
                <li>Principal Component Analysis (PCA)</li>
            </ul>
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar - Quick Metrics Summary
st.sidebar.markdown("""
    <div class="card-container" style="background-color: #F8FAFC; border-left: 4px solid #4F46E5;">
        <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: #4F46E5;">Model Summary (PR-AUC)</h4>
        <table style="width:100%; font-size: 0.85rem; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #E2E8F0; height: 30px;">
                <td><b>Random Forest</b></td>
                <td style="text-align: right; color: #059669; font-weight: bold;">0.825</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0; height: 30px;">
                <td><b>Logistic Regression</b></td>
                <td style="text-align: right; color: #4F46E5; font-weight: bold;">0.764</td>
            </tr>
            <tr style="height: 30px;">
                <td><b>Isolation Forest</b></td>
                <td style="text-align: right; color: #EA580C; font-weight: bold;">0.105</td>
            </tr>
        </table>
    </div>
""", unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 System Overview", 
    "🌌 PCA Component Analysis", 
    "📈 Model Evaluation & Comparison", 
    "⚡ Transaction Sandbox (Live Predictor)"
])

# ================= TAB 1: SYSTEM OVERVIEW =================
with tab1:
    st.markdown("### 📊 Dataset Overview & Class Imbalance")
    
    # KPI metrics row
    stats = metrics_data["dataset_stats"]
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.markdown(f"""
            <div class="card-container metric-card" style="border-top: 4px solid #636EFA;">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-val" style="color: #636EFA;">{stats['total_transactions']:,}</div>
                <div style="font-size: 0.8rem; color: #64748B;">Complete Kaggle Dataset</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
            <div class="card-container metric-card" style="border-top: 4px solid #EF553B;">
                <div class="metric-label">Fraudulent Transactions</div>
                <div class="metric-val" style="color: #EF553B;">{stats['total_fraud']:,}</div>
                <div style="font-size: 0.8rem; color: #64748B;">Class = 1</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
            <div class="card-container metric-card" style="border-top: 4px solid #00CC96;">
                <div class="metric-label">Base Fraud Rate</div>
                <div class="metric-val" style="color: #00CC96;">{stats['fraud_percentage']:.4f}%</div>
                <div style="font-size: 0.8rem; color: #64748B;">Highly Imbalanced Ratio</div>
            </div>
        """, unsafe_allow_html=True)

    # Class Imbalance Visualizer Row
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        # Pie chart showing balance
        pie_df = pd.DataFrame({
            "Transaction Type": ["Legitimate (99.83%)", "Fraudulent (0.17%)"],
            "Count": [stats['total_transactions'] - stats['total_fraud'], stats['total_fraud']]
        })
        fig_pie = px.pie(
            pie_df, 
            values='Count', 
            names='Transaction Type',
            color='Transaction Type',
            color_discrete_map={"Legitimate (99.83%)": "#00CC96", "Fraudulent (0.17%)": "#EF553B"},
            hole=0.4,
            title="Proportion of Fraudulent Transactions"
        )
        fig_pie.update_layout(
            legend=dict(orientation="h", y=-0.1),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        # Distribution of amounts boxplot
        fig_box = px.box(
            plot_df, 
            x='Class', 
            y='Amount', 
            color='Class',
            color_discrete_map={0: "#00CC96", 1: "#EF553B"},
            title="Comparison of Transaction Amounts (Log-Scaled Y)",
            points="outliers",
            labels={"Class": "Transaction Type (0 = Legit, 1 = Fraud)"}
        )
        fig_box.update_yaxes(type="log", title="Amount ($) - Log Scale")
        fig_box.update_layout(
            showlegend=False,
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
        > [!NOTE]  
        > **Why this imbalance matters:** In a system with 99.83% legitimate transactions, a dummy model that predicts "Legitimate" for every transaction achieves **99.83% accuracy**. However, it detects **0% of fraud**. This is why accuracy is a dangerous metric for fraud detection. We must optimize for **Precision** (avoiding false alarms) and **Recall** (catching all fraud).
    """)

# ================= TAB 2: PCA COMPONENT ANALYSIS =================
with tab2:
    st.markdown("### 🌌 Principal Component Analysis (PCA)")
    st.write(
        "Because credit card transaction features are highly sensitive, Kaggle's raw dataset contains only PCA-transformed components ($V_1$ to $V_{28}$). "
        "These components represent the directions of maximum variance in the original feature space. Below, you can explore how these components separate fraud from legitimate transactions."
    )
    
    col_pca_ctrl, col_pca_plot = st.columns([1, 2])
    
    with col_pca_ctrl:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Projection Controls")
        
        # User dropdown selection for X and Y components
        pca_features = [f"V{i}" for i in range(1, 29)]
        x_pca = st.selectbox("Select X-Axis Component:", pca_features, index=0) # V1
        y_pca = st.selectbox("Select Y-Axis Component:", pca_features, index=1) # V2
        
        st.markdown("""
            **Educational Guide:**
            * Look for clusters or separation boundaries where fraud points (red) group together.
            * Features like $V_{17}$, $V_{14}$, and $V_{12}$ show strong class separation, whereas other features are highly overlapping.
            * Hover over points in the scatter plot to see transaction amount details.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_pca_plot:
        # Import plotting functions
        from src.utils import plot_pca_separation_2d
        fig_pca = plot_pca_separation_2d(plot_df, x_col=x_pca, y_col=y_pca)
        st.plotly_chart(fig_pca, use_container_width=True)

# ================= TAB 3: MODEL PERFORMANCE & EVALUATION =================
with tab3:
    st.markdown("### 📈 Machine Learning Model Comparison")
    st.write(
        "Here we evaluate three models: **Logistic Regression** (with class-balance weighting), **Random Forest** (ensemble tree), and **Isolation Forest** (unsupervised anomaly detection)."
    )
    
    # Model evaluation metrics table
    perf_data = {
        "Metric": ["Precision (PPV)", "Recall (Sensitivity)", "F1-Score (Harmonic Mean)", "PR-AUC (AUPRC)", "ROC-AUC"],
        "Logistic Regression": [
            f"{metrics_data['logistic_regression']['precision']:.4f}",
            f"{metrics_data['logistic_regression']['recall']:.4f}",
            f"{metrics_data['logistic_regression']['f1']:.4f}",
            f"{metrics_data['logistic_regression']['pr_auc']:.4f}",
            f"{metrics_data['logistic_regression']['roc_auc']:.4f}"
        ],
        "Random Forest": [
            f"{metrics_data['random_forest']['precision']:.4f}",
            f"{metrics_data['random_forest']['recall']:.4f}",
            f"{metrics_data['random_forest']['f1']:.4f}",
            f"{metrics_data['random_forest']['pr_auc']:.4f}",
            f"{metrics_data['random_forest']['roc_auc']:.4f}"
        ],
        "Isolation Forest (Anomaly)": [
            f"{metrics_data['isolation_forest']['precision']:.4f}",
            f"{metrics_data['isolation_forest']['recall']:.4f}",
            f"{metrics_data['isolation_forest']['f1']:.4f}",
            f"{metrics_data['isolation_forest']['pr_auc']:.4f}",
            f"{metrics_data['isolation_forest']['roc_auc']:.4f}"
        ]
    }
    
    perf_df = pd.DataFrame(perf_data)
    st.table(perf_df)
    
    # Side-by-side Curves
    col_curve1, col_curve2 = st.columns(2)
    
    # Load evaluation curve vectors
    y_test_sample = np.array(metrics_data["y_test_sample"])
    y_probs_sample = {
        "Logistic Regression": np.array(metrics_data["y_probs_sample"]["logistic_regression"]),
        "Random Forest": np.array(metrics_data["y_probs_sample"]["random_forest"]),
        "Isolation Forest": np.array(metrics_data["y_probs_sample"]["isolation_forest"])
    }
    
    with col_curve1:
        from src.utils import plot_precision_recall_curve
        fig_pr = plot_precision_recall_curve(y_test_sample, y_probs_sample, list(y_probs_sample.keys()))
        st.plotly_chart(fig_pr, use_container_width=True)
        
    with col_curve2:
        from src.utils import plot_roc_curve
        fig_roc = plot_roc_curve(y_test_sample, y_probs_sample, list(y_probs_sample.keys()))
        st.plotly_chart(fig_roc, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 🧬 Confusion Matrix Heatmaps")
    
    col_cm1, col_cm2, col_cm3 = st.columns(3)
    from src.utils import plot_confusion_matrix
    
    # Convert saved confusion matrix dictionaries back to values
    with col_cm1:
        cm_lr_data = metrics_data["logistic_regression"]["confusion_matrix"]
        # build 2x2 confusion matrix array
        cm_lr = np.array([[cm_lr_data["tn"], cm_lr_data["fp"]], [cm_lr_data["fn"], cm_lr_data["tp"]]])
        
        fig_cm_lr = go.Figure(data=go.Heatmap(
            z=cm_lr, x=['Legit (Pred)', 'Fraud (Pred)'], y=['Legit (Actual)', 'Fraud (Actual)'],
            text=[[str(val) for val in row] for row in cm_lr], texttemplate="%{text}",
            colorscale='Purples', showscale=False
        ))
        fig_cm_lr.update_layout(title="Logistic Regression CM", width=320, height=320, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_cm_lr, use_container_width=True)
        
    with col_cm2:
        cm_rf_data = metrics_data["random_forest"]["confusion_matrix"]
        cm_rf = np.array([[cm_rf_data["tn"], cm_rf_data["fp"]], [cm_rf_data["fn"], cm_rf_data["tp"]]])
        
        fig_cm_rf = go.Figure(data=go.Heatmap(
            z=cm_rf, x=['Legit (Pred)', 'Fraud (Pred)'], y=['Legit (Actual)', 'Fraud (Actual)'],
            text=[[str(val) for val in row] for row in cm_rf], texttemplate="%{text}",
            colorscale='Blues', showscale=False
        ))
        fig_cm_rf.update_layout(title="Random Forest CM", width=320, height=320, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        
    with col_cm3:
        cm_if_data = metrics_data["isolation_forest"]["confusion_matrix"]
        cm_if = np.array([[cm_if_data["tn"], cm_if_data["fp"]], [cm_if_data["fn"], cm_if_data["tp"]]])
        
        fig_cm_if = go.Figure(data=go.Heatmap(
            z=cm_if, x=['Legit (Pred)', 'Fraud (Pred)'], y=['Legit (Actual)', 'Fraud (Actual)'],
            text=[[str(val) for val in row] for row in cm_if], texttemplate="%{text}",
            colorscale='Oranges', showscale=False
        ))
        fig_cm_if.update_layout(title="Isolation Forest CM", width=320, height=320, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_cm_if, use_container_width=True)
        
    st.markdown("""
        > [!TIP]  
        > **Key Takeaway:**
        > * **Logistic Regression** (supervised balanced class weights) has a **91.8% Recall**, catching 90 out of 98 fraud cases in the test set. However, because it is biased to predict fraud, it generates **1,416 false positives** (low precision: 6%). In retail banking, this results in high customer irritation due to blocked cards.
        > * **Random Forest** manages to catch **80 out of 98 frauds** (81.6% Recall) but with only **21 false positives** (79.2% Precision). This represents a highly practical balance.
        > * **Isolation Forest** (unsupervised) operates without labels and identifies outliers. While its metrics are lower, it is vital because it can detect *novel fraud patterns (zero-day attacks)* that supervised models trained on past fraud profiles would miss completely.
    """)

# ================= TAB 4: TRANSACTION SANDBOX (PREDICTOR) =================
with tab4:
    st.markdown("### ⚡ Transaction Sandbox & Live Predictor")
    st.write(
        "Simulate credit card transaction details below. You can load preset templates (such as legitimate or fraudulent transactions) "
        "or adjust individual feature values manually to see how the models respond in real time."
    )
    
    # 4.1 Preset Scenarios
    col_presets = st.columns(4)
    preset_clicked = None
    
    # Load representative sample transactions from plot_df
    # We find a true fraud row and a true normal row
    fraud_samples = plot_df[plot_df['Class'] == 1]
    normal_samples = plot_df[plot_df['Class'] == 0]
    
    sample_fraud_row = fraud_samples.iloc[0] if len(fraud_samples) > 0 else None
    sample_normal_row = normal_samples.iloc[0] if len(normal_samples) > 0 else None
    
    with col_presets[0]:
        if st.button("🟢 Scenario A: Normal Legitimate Transaction", use_container_width=True):
            preset_clicked = "normal"
            
    with col_presets[1]:
        if st.button("🔴 Scenario B: Suspicious Transaction (Class 1 Sample)", use_container_width=True):
            preset_clicked = "fraud"
            
    with col_presets[2]:
        if st.button("⚠️ Scenario C: High Amount Anomaly", use_container_width=True):
            preset_clicked = "high_amount"
            
    with col_presets[3]:
        if st.button("🌌 Scenario D: High-Risk Outlier Profile", use_container_width=True):
            preset_clicked = "outlier_profile"

    # Set default values based on selected scenario
    default_vals = {}
    
    # Default PCA values (V1-V28)
    for i in range(1, 29):
        default_vals[f"V{i}"] = 0.0
        
    default_vals["Time"] = 50000.0
    default_vals["Amount"] = 120.0
    
    if preset_clicked == "normal" and sample_normal_row is not None:
        default_vals["Time"] = float(sample_normal_row["Time"]) if "Time" in sample_normal_row else 45000.0
        default_vals["Amount"] = float(sample_normal_row["Amount"]) if "Amount" in sample_normal_row else 50.0
        for i in range(1, 29):
            default_vals[f"V{i}"] = float(sample_normal_row[f"V{i}"])
            
    elif preset_clicked == "fraud" and sample_fraud_row is not None:
        default_vals["Time"] = float(sample_fraud_row["Time"]) if "Time" in sample_fraud_row else 65000.0
        default_vals["Amount"] = float(sample_fraud_row["Amount"]) if "Amount" in sample_fraud_row else 850.00
        for i in range(1, 29):
            default_vals[f"V{i}"] = float(sample_fraud_row[f"V{i}"])
            
    elif preset_clicked == "high_amount":
        # High amount normal transaction
        default_vals["Time"] = 80000.0
        default_vals["Amount"] = 25000.0 # Extreme Amount
        # Normal average PCA values (mostly zeroes)
        for i in range(1, 29):
            default_vals[f"V{i}"] = 0.0
            
    elif preset_clicked == "outlier_profile" and sample_fraud_row is not None:
        default_vals["Time"] = 10000.0
        default_vals["Amount"] = 4500.0
        # Give highly skewed features typical of fraud
        for i in range(1, 29):
            # Scale up fraud component features to make it a heavy outlier
            default_vals[f"V{i}"] = float(sample_fraud_row[f"V{i}"]) * 1.5

    # 4.2 Interactive Input Layout
    st.markdown("<div class='card-container'>", unsafe_allow_html=True)
    st.markdown("#### 📝 Edit Transaction Profile")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        time_input = st.number_input(
            "Transaction Time (seconds since first record):", 
            min_value=0.0, 
            max_value=200000.0, 
            value=default_vals["Time"],
            step=100.0
        )
        
    with col_input2:
        amount_input = st.number_input(
            "Transaction Amount ($):", 
            min_value=0.0, 
            max_value=100000.0, 
            value=default_vals["Amount"],
            step=5.0
        )
        
    # Top 6 features of importance in credit card fraud detection are typically:
    # V17, V14, V12, V10, V16, V11
    # We display sliders for these 6 features explicitly, and collapse the other 22 into an advanced container!
    st.write("---")
    st.write("✏️ **Key PCA Anomaly Components:** (Adjust to shift transaction probability)")
    
    key_features = ["V17", "V14", "V12", "V10", "V16", "V11"]
    col_keys = st.columns(3)
    
    pca_inputs = {}
    for idx, feat in enumerate(key_features):
        col_idx = idx % 3
        with col_keys[col_idx]:
            pca_inputs[feat] = st.slider(
                f"Feature {feat}:", 
                min_value=-30.0, 
                max_value=30.0, 
                value=default_vals[feat],
                step=0.1,
                help=f"High-impact PCA feature {feat}. Fraud profiles often show values skewed from zero."
            )
            
    with st.expander("🌐 Advanced: View/Modify other 22 PCA Components"):
        col_adv = st.columns(4)
        adv_idx = 0
        for i in range(1, 29):
            feat_name = f"V{i}"
            if feat_name not in key_features:
                adv_col_idx = adv_idx % 4
                with col_adv[adv_col_idx]:
                    pca_inputs[feat_name] = st.number_input(
                        f"Feature {feat_name}:", 
                        min_value=-50.0, 
                        max_value=50.0, 
                        value=default_vals[feat_name],
                        step=0.05
                    )
                adv_idx += 1
                
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 4.3 Model Prediction Inference
    # Create input DataFrame
    input_data = {}
    input_data['Time'] = [time_input]
    input_data['Amount'] = [amount_input]
    for i in range(1, 29):
        input_data[f"V{i}"] = [pca_inputs[f"V{i}"]]
        
    input_df = pd.DataFrame(input_data)
    
    # Preprocess
    if isinstance(preprocessor, dict):
        time_scaler = preprocessor['time_scaler']
        amount_scaler = preprocessor['amount_scaler']
        processed_input_df = input_df.copy()
        processed_input_df['scaled_time'] = time_scaler.transform(processed_input_df[['Time']])
        processed_input_df['scaled_amount'] = amount_scaler.transform(processed_input_df[['Amount']])
        processed_input_df = processed_input_df.drop(['Time', 'Amount'], axis=1)
    else:
        processed_input_df = preprocessor.transform(input_df)
    
    # Run predictions
    # Models expect features in a specific order: V1 to V28, scaled_time, scaled_amount
    cols_order = [f"V{i}" for i in range(1, 29)] + ['scaled_time', 'scaled_amount']
    model_input_df = processed_input_df[cols_order]
    
    lr_pred = int(lr_model.predict(model_input_df)[0])
    lr_prob = float(lr_model.predict_proba(model_input_df)[0][1])
    
    rf_pred = int(rf_model.predict(model_input_df)[0])
    rf_prob = float(rf_model.predict_proba(model_input_df)[0][1])
    
    if_pred = int(if_model.predict(model_input_df)[0])
    if_score = float(if_model.predict_proba(model_input_df)[0][1]) # Anomaly score
    
    # Render predictions
    st.markdown("### 🔮 Live Model Predictions")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>Logistic Regression</h4>", unsafe_allow_html=True)
        if lr_pred == 1:
            st.markdown("<span class='badge badge-fraud'>❌ FRAUD DETECTED</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-legit'>🟢 LEGITIMATE</span>", unsafe_allow_html=True)
        st.metric(label="Fraud Confidence (Prob)", value=f"{lr_prob*100:.2f}%")
        st.markdown("<p style='font-size:0.8rem; color:#64748B;'>High recall bias, low tolerance for suspect profiles.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_res2:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>Random Forest</h4>", unsafe_allow_html=True)
        if rf_pred == 1:
            st.markdown("<span class='badge badge-fraud'>❌ FRAUD DETECTED</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-legit'>🟢 LEGITIMATE</span>", unsafe_allow_html=True)
        st.metric(label="Fraud Confidence (Prob)", value=f"{rf_prob*100:.2f}%")
        st.markdown("<p style='font-size:0.8rem; color:#64748B;'>High precision bias, evaluates complex decision limits.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_res3:
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>Isolation Forest</h4>", unsafe_allow_html=True)
        if if_pred == 1:
            st.markdown("<span class='badge badge-fraud'>⚠️ ANOMALOUS TRANSACTION</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-legit'>🟢 NORMAL TRANSACTION</span>", unsafe_allow_html=True)
        st.metric(label="Anomaly Score Intensity", value=f"{if_score*100:.2f}%")
        st.markdown("<p style='font-size:0.8rem; color:#64748B;'>Unsupervised. Computes isolate distance score.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Overall Consensus diagnosis
    votes = sum([lr_pred, rf_pred, if_pred])
    st.markdown("---")
    st.markdown("#### 🚨 Consensus Fraud Assessment")
    
    if votes == 3:
        st.markdown("""
            <div style='background-color:#FEE2E2; border-left:6px solid #EF553B; padding:1.5rem; border-radius:8px;'>
                <h4 style='margin:0; color:#991B1B;'>🔴 High Alert: Critical Risk of Fraud</h4>
                <p style='margin:0.5rem 0 0 0; color:#7F1D1D;'>
                    All three models (supervised and unsupervised anomaly detection) agree that this transaction is highly anomalous and matches fraud profile patterns. Automated block recommended.
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif votes == 2:
        st.markdown("""
            <div style='background-color:#FEF3C7; border-left:6px solid #D97706; padding:1.5rem; border-radius:8px;'>
                <h4 style='margin:0; color:#92400E;'>🟡 Medium Alert: Suspect Transaction</h4>
                <p style='margin:0.5rem 0 0 0; color:#78350F;'>
                    Two models flag this transaction. Typically, Logistic Regression and Random Forest or Isolation Forest identify it. Manual review or secondary authentication (OTP check) recommended.
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif votes == 1:
        st.markdown("""
            <div style='background-color:#EFF6FF; border-left:6px solid #2563EB; padding:1.5rem; border-radius:8px;'>
                <h4 style='margin:0; color:#1E40AF;'>🔵 Low Alert: Minor Anomaly / Soft Warning</h4>
                <p style='margin:0.5rem 0 0 0; color:#1E3A8A;'>
                    A single model (often the high-recall Logistic Regression or the anomaly-based Isolation Forest) flagged this. This is a low-risk transaction. Keep monitoring or pass silently.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background-color:#ECFDF5; border-left:6px solid #10B981; padding:1.5rem; border-radius:8px;'>
                <h4 style='margin:0; color:#065F46;'>🟢 Safe: Clear Transaction</h4>
                <p style='margin:0.5rem 0 0 0; color:#064E3B;'>
                    All models agree that this transaction is legitimate and follows normal patterns. No action required.
                </p>
            </div>
        """, unsafe_allow_html=True)
