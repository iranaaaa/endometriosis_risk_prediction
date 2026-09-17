import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def train_xgboost(X_train, y_train):
    """Train the XGBoost classifier."""
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred):
    """Generate confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low Risk', 'High Risk'], 
                yticklabels=['Low Risk', 'High Risk'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_test, y_prob):
    """Generate ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def explain_model_shap(model, X_train, X_test):
    """Generate SHAP explainer and values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

def plot_shap_summary(shap_values, X_test):
    """Generate SHAP summary plot (feature importance)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig
