import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

def save_artifact(artifact, filename, directory="models"):
    """
    Serializes a model, scaler, or preprocessor to the specified directory.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    print(f"Saving artifact to {filepath}...")
    joblib.dump(artifact, filepath)
    print("Artifact saved successfully.")

def load_artifact(filename, directory="models"):
    """
    Deserializes a model, scaler, or preprocessor from the specified directory.
    """
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    print(f"Loading artifact from {filepath}...")
    return joblib.load(filepath)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Generates an interactive Plotly Heatmap for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Define labels and annotations
    x_labels = ['Predicted Legitimate', 'Predicted Fraudulent']
    y_labels = ['Actual Legitimate', 'Actual Fraudulent']
    
    # Text annotations in each box
    z_text = [[str(val) for val in row] for row in cm]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        text=z_text,
        texttemplate="%{text}",
        hoverongaps=False,
        colorscale='Viridis',
        showscale=False
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=450,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, Roboto, sans-serif", size=12)
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_probs, model_names):
    """
    Generates an interactive Plotly line chart comparing Precision-Recall curves.
    y_probs should be a dictionary: {model_name: probabilities_array}
    """
    fig = go.Figure()
    
    # Plot baseline random classifier line
    # Precision baseline = fraction of positive class
    pos_fraction = np.sum(y_true) / len(y_true)
    fig.add_shape(
        type='line', line=dict(dash='dash', color='gray', width=1.5),
        x0=0, x1=1, y0=pos_fraction, y1=pos_fraction
    )
    fig.add_annotation(
        x=0.85, y=pos_fraction + 0.05,
        text=f"No-Skill Baseline ({pos_fraction*100:.2f}%)",
        showarrow=False,
        font=dict(color='gray', size=10)
    )
    
    colors = ['#636EFA', '#EF553B', '#00CC96']
    for idx, (name, probs) in enumerate(y_probs.items()):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        # Compute AUPRC
        from sklearn.metrics import auc
        pr_auc = auc(recall, precision)
        
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f"{name} (AUPRC = {pr_auc:.3f})",
            line=dict(color=color, width=2.5)
        ))
        
    fig.update_layout(
        title="Precision-Recall Curves Comparison",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        xaxis=dict(range=[0.0, 1.05]),
        yaxis=dict(range=[0.0, 1.05]),
        legend=dict(x=0.02, y=0.02, bgcolor='rgba(255,255,255,0.7)'),
        height=500,
        font=dict(family="Inter, Roboto, sans-serif")
    )
    
    return fig

def plot_roc_curve(y_true, y_probs, model_names):
    """
    Generates an interactive Plotly line chart comparing ROC curves.
    y_probs should be a dictionary: {model_name: probabilities_array}
    """
    fig = go.Figure()
    
    # Diagonal line (random guessing baseline)
    fig.add_shape(
        type='line', line=dict(dash='dash', color='gray', width=1.5),
        x0=0, x1=1, y0=0, y1=1
    )
    
    colors = ['#636EFA', '#EF553B', '#00CC96']
    for idx, (name, probs) in enumerate(y_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, probs)
        # Compute ROC AUC
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, probs)
        
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{name} (ROC AUC = {roc_auc:.3f})",
            line=dict(color=color, width=2.5)
        ))
        
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall)",
        xaxis=dict(range=[0.0, 1.05]),
        yaxis=dict(range=[0.0, 1.05]),
        legend=dict(x=0.6, y=0.1, bgcolor='rgba(255,255,255,0.7)'),
        height=500,
        font=dict(family="Inter, Roboto, sans-serif")
    )
    
    return fig

def plot_pca_separation_2d(df, x_col='V1', y_col='V2', sample_size=10000):
    """
    Generates a 2D scatter plot demonstrating classification separation using PCA components.
    Since fraud is very rare, we downsample the normal transactions but retain ALL fraud cases
    to ensure both are visible and properly compared.
    """
    fraud_df = df[df['Class'] == 1]
    normal_df = df[df['Class'] == 0]
    
    # Downsample normal transactions
    num_normal_samples = min(len(normal_df), sample_size - len(fraud_df))
    if num_normal_samples > 0:
        normal_sampled = normal_df.sample(n=num_normal_samples, random_state=42)
        plot_df = pd.concat([normal_sampled, fraud_df])
    else:
        plot_df = df
        
    # Map Class labels for readability in the chart legend
    plot_df = plot_df.copy()
    plot_df['Transaction Type'] = plot_df['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color='Transaction Type',
        color_discrete_map={'Legitimate': '#00CC96', 'Fraudulent': '#EF553B'},
        opacity=0.6,
        title=f"PCA Projection: {x_col} vs {y_col}",
        labels={x_col: x_col, y_col: y_col},
        hover_data=['Amount'] if 'Amount' in plot_df.columns else []
    )
    
    fig.update_layout(
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
        height=500,
        font=dict(family="Inter, Roboto, sans-serif")
    )
    
    return fig
