import numpy as np

from .loss import Loss


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) loss.

        Args:
            predictions (np.ndarray): The predicted values from the model.
            targets (np.ndarray): The true target values.

        Returns:
            float: The computed MSE loss value.
        """
        self.predictions: np.ndarray = predictions
        self.targets: np.ndarray = targets

        return float(np.mean((predictions - targets) ** 2))

    def backward(self) -> np.ndarray:
        """
        Compute the gradient of the MSE loss with respect to the predictions.

        Returns:
            np.ndarray: The gradient of the MSE loss.
        """
        return (2 / self.targets.shape[0]) * (self.predictions - self.targets)
