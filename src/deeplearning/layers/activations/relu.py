import numpy as np

from ..layer import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation layer.
    """

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the ReLU activation function.

        Args:
            tensor (np.ndarray): Input data to the layer.

        Returns:
            np.ndarray: Activated tensor.
        """
        self.mask: np.ndarray = tensor > 0

        return tensor * self.mask

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the ReLU activation function.

        Args:
            gradient (np.ndarray): Gradient.

        Returns:
            np.ndarray: Gradient.
        """
        return gradient * self.mask
