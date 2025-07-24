from sklearn.metrics import matthews_corrcoef, accuracy_score
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, pearsonr

def rsa(distance_matrix1, distance_matrix2):
    ''' compute the RSA between two distance matrices.
    the distance matrices are expected to be square matrices of shape
        (n_classes, n_classes).
    the upper triangle of the distance matrices are extracted,
    and then the RSA is computed with rank correlation.
    '''
    upper_triangle1 = get_upper_triangle_values(distance_matrix1)
    upper_triangle2 = get_upper_triangle_values(distance_matrix2)
    spearman, _ = pearsonr(upper_triangle1, upper_triangle2)
    return spearman

def confusion_matrix_to_rsa(confusion_matrix1, confusion_matrix2):
    ''' compute the RSA between two confusion matrices.
    the confusion matrices are expected to be square matrices of shape 
        (n_classes, n_classes).
    the confusion matrices converted to JSD matrices, 
        and then the RSA is computed with rank correlation.
    '''
    jsd1 = confusion_matrix_to_jsd_matrix(confusion_matrix1)
    jsd2 = confusion_matrix_to_jsd_matrix(confusion_matrix2)
    return rsa(jsd1, jsd2)

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


def get_upper_triangle_values(matrix):
    """
    Extracts the upper triangle values of a square matrix.
    
    Parameters:
    - matrix: 2D numpy array or list of lists representing a square matrix.
    
    Returns:
    - upper_triangle_values: 1D numpy array containing the upper triangle values.
    excluding the diagonal if k=1.
    """
    return matrix[np.triu_indices_from(matrix, k=1)]

