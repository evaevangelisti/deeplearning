from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    def __init__(self, parameters: list[np.ndarray]):
        """
        Initialize the optimizer.

        Args:
            parameters (list[np.ndarray]): List of model parameters.
        """
        self.parameters: list[np.ndarray] = parameters

    @abstractmethod
    def step(self, gradients: list[np.ndarray]) -> None:
        """
        Perform a single optimization step.

        Args:
            gradients (list[np.ndarray]): Gradients.
        """
        pass
