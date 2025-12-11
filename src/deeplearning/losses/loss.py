from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the loss value given predictions and targets.

        Args:
            predictions (np.ndarray): The predicted values from the model.
            targets (np.ndarray): The true target values.

        Returns:
            float: The computed loss value.
        """
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.

        Returns:
            np.ndarray: The gradient of the loss.
        """
        pass
