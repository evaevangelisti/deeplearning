import numpy as np


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions against the target labels.

    Args:
        predictions (np.ndarray): The predicted outputs from the model.
        targets (np.ndarray): The true target labels.

    Returns:
        float: The accuracy as a float value between 0 and 1.
    """
    predicted_classes: np.ndarray = np.argmax(predictions, axis=1)
    target_classes: np.ndarray = (
        np.argmax(targets, axis=1) if targets.ndim == 2 else targets
    )

    return float(np.mean(predicted_classes == target_classes))
