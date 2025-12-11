import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute the softmax probabilities from logits.

    Args:
        logits (np.ndarray): Logits.

    Returns:
        np.ndarray: Softmax probabilities.
    """
    shifted_logits: np.ndarray = logits - np.max(logits, axis=1, keepdims=True)
    exponential_logits: np.ndarray = np.exp(shifted_logits)

    return exponential_logits / np.sum(exponential_logits, axis=1, keepdims=True)
