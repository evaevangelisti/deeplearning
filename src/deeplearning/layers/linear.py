import numpy as np

from .layer import Layer


class Linear(Layer):
    """
    Fully connected linear layer.
    """

    def __init__(self, input_features: int, output_features: int):
        """
        Initialize the linear layer with random weights and zero biases.

        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output features.
        """
        self.weights: np.ndarray = np.random.randn(input_features, output_features)
        self.weights *= 0.01

        self.biases: np.ndarray = np.zeros((1, output_features))

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the linear layer.

        Args:
            tensor (np.ndarray): Input data to the layer.

        Returns:
            np.ndarray: Output data from the layer.
        """
        self.tensor: np.ndarray = tensor

        return tensor @ self.weights + self.biases

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the linear layer.

        Args:
            gradient (np.ndarray): Gradient.

        Returns:
            np.ndarray: Gradient.
        """
        self.weights_gradient: np.ndarray = self.tensor.T @ gradient
        self.biases_gradient: np.ndarray = np.sum(gradient, axis=0, keepdims=True)

        return gradient @ self.weights.T
