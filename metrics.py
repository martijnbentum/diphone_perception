from sklearn.metrics import matthews_corrcoef, accuracy_score
import numpy as np
from scipy.spatial.distance import jensenshannon

def confusion_matrix_to_jsd_matrix(confusion_matrix):
    """
    Computes the Jensen-Shannon distance matrix from a confusion matrix.
    
    Parameters:
    - confusion_matrix: square confusion matrix of shape (n_classes, n_classes)

    Returns:
    - jsd_matrix: symmetric distance matrix (n_classes, n_classes)
    """
    # Normalize each row to get class-conditional probability distributions
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    prob_matrix = confusion_matrix / (row_sums + 1e-12) # avoid division by zero

    n = prob_matrix.shape[0]
    jsd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = jensenshannon(prob_matrix[i], prob_matrix[j])
            jsd_matrix[i, j] = dist
            jsd_matrix[j, i] = dist  # symmetry

    return jsd_matrix


def mcc(y_true, y_pred):
    """
    Calculate the Matthews correlation coefficient (MCC).
    
    Parameters:
    y_true (list): Ground truth (correct) labels.
    y_pred (list): Predicted labels.
    
    Returns:
    float: MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy score.
    
    Parameters:
    y_true (list): Ground truth (correct) labels.
    y_pred (list): Predicted labels.
    
    Returns:
    float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def confusion_matrix_to_mcc(cm):
    y_true, y_pred = confusion_matrix_to_gt_pred(cm)
    return mcc(y_true, y_pred)

def confusion_matrix_accuracy(cm):
    y_true, y_pred = confusion_matrix_to_gt_pred(cm)
    return accuracy(y_true, y_pred)

def confusion_matrix_to_gt_pred(cm):
    """
    Convert a confusion matrix to ground truth and predicted labels.
    
    Parameters:
    cm (list of list of int): Confusion matrix.
    
    Returns:
    tuple: (ground_truth, predicted) where both are lists of labels.
    """
    ground_truth = []
    predicted = []
    
    for i, row in enumerate(cm):
        for j, count in enumerate(row):
            ground_truth.extend([i] * count)
            predicted.extend([j] * count)
    
    return ground_truth, predicted

