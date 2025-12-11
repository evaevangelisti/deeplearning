import numpy as np

from ..utils.softmax import softmax
from .loss import Loss


class CrossEntropy(Loss):
    """
    Cross-Entropy loss function.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the Cross-Entropy loss.

        Args:
            predictions (np.ndarray): The predicted logits from the model.
            targets (np.ndarray): The true target class indices.

        Returns:
            float: The computed Cross-Entropy loss value.
        """
        self.predictions: np.ndarray = predictions
        self.targets: np.ndarray = targets

        self.probabilities: np.ndarray = softmax(predictions)

        batch_indices: np.ndarray = np.arange(targets.shape[0])
        target_probabilities: np.ndarray = self.probabilities[batch_indices, targets]

        losses: np.ndarray = -np.log(target_probabilities)

        return float(np.mean(losses))

    def backward(self) -> np.ndarray:
        """
        Compute the gradient of the Cross-Entropy loss with respect to the predictions.

        Returns:
            np.ndarray: The gradient of the Cross-Entropy loss.
        """
        gradient: np.ndarray = self.probabilities.copy()
        batch_size: int = self.targets.shape[0]

        gradient[np.arange(batch_size), self.targets] -= 1
        gradient /= batch_size

        return gradient
