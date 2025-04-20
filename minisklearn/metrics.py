import numpy as np

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    error = (y_pred == y_true)
    return np.sum(error)/len(y_true)

def precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    error = (y_pred == y_true)
    TP = np.sum(y_pred[error])
    FP = np.sum(y_pred[~error])

    return TP / (TP + FP)

def recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    error = (y_pred == y_true)
    TP = np.sum(y_pred[error])
    FN = np.sum(1-y_pred[~error])
    return TP / (TP + FN)

def f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)

    return 2 * p * r / (p + r)

def specificity(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    error = (y_pred == y_true)
    TN = np.sum(1-y_pred[error])
    FP = np.sum(y_pred[~error])

    return TN / (TN + FP)