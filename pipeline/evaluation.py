from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score
)
import numpy as np

def evaluate_clustering(true_labels, predicted_labels, exclude_noise=True):
    """
    Evaluates clustering performance using several external cluster validity indices.
    
    Parameters:
        true_labels (array-like): Ground-truth cluster ids.
        predicted_labels (array-like): Predicted cluster labels from the clustering algorithm.
        exclude_noise (bool): If True, removes noise points (where predicted label == -1) from evaluation.
    
    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    if exclude_noise:
        mask = predicted_labels != -1
        true_labels = true_labels[mask]
        predicted_labels = predicted_labels[mask]

    metrics = {
        "Adjusted_Rand_Index": adjusted_rand_score(true_labels, predicted_labels),
        "Adjusted_Mutual_Info": adjusted_mutual_info_score(true_labels, predicted_labels),
        "Homogeneity": homogeneity_score(true_labels, predicted_labels),
        "Completeness": completeness_score(true_labels, predicted_labels),
        "V_measure": v_measure_score(true_labels, predicted_labels),
        "Fowlkes_Mallows": fowlkes_mallows_score(true_labels, predicted_labels)
    }
    return metrics
