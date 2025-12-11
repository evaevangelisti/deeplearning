import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(
        self,
        parameters: list[np.ndarray],
        learning_rate: float = 0.001,
        momentum_decay: float = 0.9,
        velocity_decay: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize the Adam optimizer.

        Args:
            parameters (list[np.ndarray]): List of model parameters.
            learning_rate (float): Learning rate. Default to 0.001.
            momentum_decay (float): Decay rate for the first moment estimates. Default to 0.9.
            velocity_decay (float): Decay rate for the second moment estimates. Default to 0.999.
            epsilon (float): Small constant to prevent division by zero. Default to 1e-8.
        """
        super().__init__(parameters)

        self.learning_rate: float = learning_rate
        self.momentum_decay: float = momentum_decay
        self.velocity_decay: float = velocity_decay
        self.epsilon: float = epsilon

        self.momentum: list[np.ndarray] = [
            np.zeros_like(parameter) for parameter in parameters
        ]

        self.velocity: list[np.ndarray] = [
            np.zeros_like(parameter) for parameter in parameters
        ]

        self.steps: int = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        """
        Perform a single optimization step using the Adam algorithm.

        Args:
            gradients (list[np.ndarray]): Gradients.
        """
        self.steps += 1

        for i, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            self.momentum[i] = (
                self.momentum_decay * self.momentum[i]
                + (1 - self.momentum_decay) * gradient
            )

            self.velocity[i] = self.velocity_decay * self.velocity[i] + (
                1 - self.velocity_decay
            ) * (gradient**2)

            momentum_corrected: np.ndarray = self.momentum[i] / (
                1 - self.momentum_decay**self.steps
            )

            velocity_corrected: np.ndarray = self.velocity[i] / (
                1 - self.velocity_decay**self.steps
            )

            parameter -= (
                self.learning_rate
                * momentum_corrected
                / (np.sqrt(velocity_corrected) + self.epsilon)
            )
