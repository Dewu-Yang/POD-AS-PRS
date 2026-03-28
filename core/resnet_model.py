
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    
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
