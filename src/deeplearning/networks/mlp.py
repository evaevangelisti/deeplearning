from ..layers.activations.relu import ReLU
from ..layers.layer import Layer
from ..layers.linear import Linear
from .neural_network import NeuralNetwork


class MLP(NeuralNetwork):
    """
    Multi-layer perceptron (MLP) neural network.
    """

    def __init__(
        self,
        input_features: int,
        hidden_layers: list[int],
        output_features: int,
    ):
        """
        Initialize a multi-layer perceptron (MLP) neural network.

        Args:
            input_features (int): Number of input features.
            hidden_layers (list[int]): List containing the number of units in each hidden layer.
            output_features (int): Number of output features.
        """
        layers: list[Layer] = []
        previous_features: int = input_features

        for hidden_units in hidden_layers:
            layers.append(Linear(previous_features, hidden_units))
            layers.append(ReLU())

            previous_features = hidden_units

        layers.append(Linear(previous_features, output_features))

        super().__init__(layers)
