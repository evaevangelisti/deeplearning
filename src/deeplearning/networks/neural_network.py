import numpy as np

from ..layers.layer import Layer


class NeuralNetwork:
    """
    A simple feedforward neural network composed of multiple layers.
    """

    def __init__(self, layers: list[Layer]):
        """
        Initialize the neural network with a list of layers.

        Args:
            layers (list[Layer]): List of layers to include in the network.
        """
        self.layers: list[Layer] = layers

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through all layers of the network.

        Args:
            tensor (np.ndarray): Input data to the network.

        Returns:
            np.ndarray: Output data from the network.
        """
        for layer in self.layers:
            tensor = layer.forward(tensor)

        return tensor

    def backward(self, gradient: np.ndarray) -> None:
        """
        Perform the backward pass through all layers of the network.

        Args:
            gradient (np.ndarray): Gradient.
        """
        for layer in self.layers:
            gradient = layer.backward(gradient)
