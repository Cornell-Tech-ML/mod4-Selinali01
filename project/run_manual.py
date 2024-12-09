"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import random
from typing import Tuple, List
import minitorch


class Network(minitorch.Module):
    """
    A simple neural network with one linear layer followed by a sigmoid activation.
    """
    def __init__(self):
        """
        Initialize the Network with a single Linear layer.
        """
        super().__init__()
        self.linear = Linear(2, 1)

    def forward(self, x: Tuple[float, float]) -> float:
        """
        Perform a forward pass through the network.

        Args:
        ----
            x: Input tensor

        Returns:
        ------
            Output after passing through the linear layer and sigmoid activation

        """
        y = self.linear(x)
        return minitorch.operators.sigmoid(y[0])


class Linear(minitorch.Module):
    """
    A linear (fully-connected) layer.
    """

    def __init__(self, in_size: int, out_size: int):
        """Initialize the Linear layer.

        Args:
        ----
            in_size (int): Number of input features
            out_size (int): Number of output features

        """
        super().__init__()
        random.seed(100)
        self.weights = []
        self.bias = []
        for i in range(in_size):
            weights = []
            for j in range(out_size):
                w = self.add_parameter(f"weight_{i}_{j}", 2 * (random.random() - 0.5))
                weights.append(w)
            self.weights.append(weights)
        for j in range(out_size):
            b = self.add_parameter(f"bias_{j}", 2 * (random.random() - 0.5))
            self.bias.append(b)

    def forward(self, inputs: Tuple[float, ...]) -> List[float]:
        """
        Perform a forward pass through the linear layer.

        Args:
        ----
            inputs (Tuple[float, ...]): Input tensor

        Returns:
        -------
            List[float]: Output tensor after applying the linear transformation

        """
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y


class ManualTrain:
    """
    A class for manually training the Network.
    """
    def __init__(self, hidden_layers: int):
        """
        Initialize the ManualTrain object.

        Args:
        ----
            hidden_layers (int): Number of hidden layers (not used in the current implementation)

        """
        self.model = Network()

    def run_one(self, x: Tuple[float, float]) -> float:
        """
        Run a single forward pass through the model.

        Args:
        ----
            x (Tuple[float, float]): Input tensor

        Returns:
        ------
            float: Output of the model for the given input

        """

        return self.model.forward((x[0], x[1]))
