"""
Evaluation utilities for fake review detection.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, labels=None):
    """
    Evaluate model performance with various metrics.
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        labels (list): Optional list of label names
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        labels (list): Optional list of label names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels if labels else ['Genuine', 'Fake'],
        yticklabels=labels if labels else ['Genuine', 'Fake']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt
