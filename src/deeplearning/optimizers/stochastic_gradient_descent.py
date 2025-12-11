import numpy as np

from .optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(self, parameters: list[np.ndarray], learning_rate: float = 0.01):
        """
        Initialize the SGD optimizer.

        Args:
            parameters (list[np.ndarray]): List of model parameters.
            learning_rate (float): Learning rate. Default to 0.01.
        """
        super().__init__(parameters)

        self.learning_rate: float = learning_rate

    def step(self, gradients: list[np.ndarray]) -> None:
        """
        Perform a single optimization step using SGD.

        Args:
            gradients (list[np.ndarray]): Gradients.
        """
        for parameter, gradient in zip(self.parameters, gradients):
            parameter -= self.learning_rate * gradient
