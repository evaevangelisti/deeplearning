from typing import cast

import numpy as np
from tqdm import trange

from .losses.loss import Loss
from .networks.neural_network import NeuralNetwork
from .optimizers.optimizer import Optimizer


def train(
    model: NeuralNetwork,
    features: np.ndarray,
    targets: np.ndarray,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int = 5,
    batch_size: int = 64,
) -> None:
    """
    Train a neural network model.

    Args:
        model (NeuralNetwork): The neural network model to train.
        features (np.ndarray): Input features for training.
        targets (np.ndarray): Target labels for training.
        loss (Loss): Loss function to use.
        optimizer (Optimizer): Optimizer to update model parameters.
        epochs (int): Number of training epochs. Defaults to 5.
        batch_size (int): Size of each training batch. Defaults to 64.
    """
    samples: int = features.shape[0]

    for _ in trange(epochs, desc="Epochs"):
        indices: np.ndarray = np.arange(samples)
        np.random.shuffle(indices)

        for start in range(0, samples, batch_size):
            index: np.ndarray = indices[start : start + batch_size]

            predictions: np.ndarray = model.forward(features[index])
            loss.forward(predictions, targets[index])

            gradient: np.ndarray = loss.backward()
            model.backward(gradient)

            gradients: list[np.ndarray] = [
                cast(np.ndarray, gradient)
                for layer in model.layers
                for gradient in (
                    [
                        getattr(layer, "weights_gradient", None),
                        getattr(layer, "biases_gradient", None),
                    ]
                )
                if gradient is not None
            ]

            optimizer.step(gradients)
