from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    """

    @abstractmethod
    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the layer.

        Args:
            tensor (np.ndarray): Input data to the layer.

        Returns:
            np.ndarray: Output data from the layer.
        """
        pass

    @abstractmethod
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.

        Args:
            gradient (np.ndarray): Gradient.

        Returns:
            np.ndarray: Gradient.
        """
        pass
