import numpy as np

from ..layer import Layer


class Tanh(Layer):
    """
    Tanh activation function layer.
    """

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Apply the tanh activation function to the input tensor.

        Args:
            tensor (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Activated tensor.
        """
        self.activated_tensor: np.ndarray = np.tanh(tensor)
        return self.activated_tensor

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the input tensor.

        Args:
            gradient (np.ndarray): Gradient.

        Returns:
            np.ndarray: Gradient.
        """
        return gradient * (1 - self.activated_tensor**2)
