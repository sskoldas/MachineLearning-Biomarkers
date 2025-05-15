import os
import sys
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def compute_metrics_from_cm(cm: np.ndarray):
    """
    Compute precision, recall, F1, and accuracy from a confusion matrix.
    cm[i, j] = number of samples with true class i predicted as class j.
    Returns a dict of metrics.
    """
    tn = cm[0, 0]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tp = cm[1, 1]
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-10)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "TN": float(tn),
        "FN": float(fn),
        "FP": float(fp),
        "TP": float(tp)
    }
    return metrics


def extract_feature_importance(pipeline, X_test, y_test, feature_names):
    """
    Extract feature importance from the classifier inside a pipeline.
    Uses coef_ or feature_importances_ if available, otherwise falls back to permutation importance.

    Returns a dict mapping feature names to importance scores.
    """
    classifier = pipeline.named_steps['classifier']
    importances = None

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_).flatten()
    else:
        result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring='accuracy')
        importances = result.importances_mean

    return dict(zip(feature_names, importances))
