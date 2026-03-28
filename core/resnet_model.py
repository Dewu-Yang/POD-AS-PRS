"""
Residual Network (ResNet) model for mapping POD coefficients to a scalar QoI.

All variants use fully-connected (linear) layers and are suitable for
flat 1-D POD coefficient vectors.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A single fully-connected residual block.

    Parameters
    ----------
    features : int
        Number of input (and output) features for both linear layers.
    """

    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Fully-connected ResNet for scalar regression from POD coefficients.

    Parameters
    ----------
    input_size : int
        Number of input POD coefficients.
    hidden_size : int, optional
        Width of the hidden representation (default 128).
    num_blocks : int, optional
        Number of residual blocks (default 6).
    dropout_rate : float, optional
        Dropout probability in the output head (default 0.1).
    """

    def __init__(self, input_size, hidden_size=128, num_blocks=6, dropout_rate=0.1):
        super(ResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


def create_resnet(input_size=100, hidden_size=128, num_blocks=6):
    """Create a ResNet model with the given architecture parameters.

    Parameters
    ----------
    input_size : int
        Number of input features (POD coefficients).
    hidden_size : int
        Hidden layer width.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    ResNet
    """
    return ResNet(input_size, hidden_size, num_blocks)


def resnet20(input_size=100):
    """Small ResNet: 6 residual blocks, hidden_size=128."""
    return ResNet(input_size=input_size, hidden_size=128, num_blocks=6)


def resnet32(input_size=100):
    """Medium ResNet: 10 residual blocks, hidden_size=128."""
    return ResNet(input_size=input_size, hidden_size=128, num_blocks=10)


def resnet44(input_size=100):
    """Large ResNet: 14 residual blocks, hidden_size=192."""
    return ResNet(input_size=input_size, hidden_size=192, num_blocks=14)


def resnet56(input_size=100):
    """Extra-large ResNet: 18 residual blocks, hidden_size=256."""
    return ResNet(input_size=input_size, hidden_size=256, num_blocks=18)


def resnet110(input_size=100):
    """Very large ResNet: 36 residual blocks, hidden_size=384."""
    return ResNet(input_size=input_size, hidden_size=384, num_blocks=36)


def resnet1202(input_size=100):
    """Extremely large ResNet: 100 residual blocks, hidden_size=512."""
    return ResNet(input_size=input_size, hidden_size=512, num_blocks=100)
