import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class BoardDataset(Dataset):
    """
    Dataset class for Connect 4 board states, policies, and values.
    """
    def __init__(self, data):  # data = np.array of (state, policy, value)
        self.states = data[:, 0]
        self.policies = data[:, 1]
        self.values = data[:, 2]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = np.int64(self.states[index].transpose(2, 0, 1))
        return state, self.policies[index], self.values[index]


class InitialConvLayer(nn.Module):
    """
    Initial convolutional layer to process the input board state.
    """
    def __init__(self, input_channels=3, output_channels=128, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = x.view(-1, 3, 6, 7)  # Reshape to match board dimensions
        return F.relu(self.batch_norm(self.conv(x)))


class ResidualLayer(nn.Module):
    """
    A single residual layer with two convolutional layers and skip connection.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        return F.relu(x + residual)


class OutputLayer(nn.Module):
    """
    Output layer with separate heads for policy and value predictions.
    """
    def __init__(self, board_dims=(6, 7), policy_channels=32, value_channels=3):
        super().__init__()
        self.value_conv = nn.Conv2d(128, value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_dims[0] * board_dims[1], 32)
        self.value_fc2 = nn.Linear(32, 1)

        self.policy_conv = nn.Conv2d(128, policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * board_dims[0] * board_dims[1], board_dims[1])

    def forward(self, x):
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1).exp()

        return p, v


class Connect4Net(nn.Module):
    """
    Main neural network for Connect 4, consisting of an initial layer,
    multiple residual layers, and an output layer.
    """
    def __init__(self, num_residual_layers=19):
        super().__init__()
        self.initial_layer = InitialConvLayer()
        self.residual_layers = nn.ModuleList([ResidualLayer() for _ in range(num_residual_layers)])
        self.output_layer = OutputLayer()

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.residual_layers:
            x = layer(x)
        return self.output_layer(x)


class CustomLoss(nn.Module):
    """
    Custom loss function combining value and policy losses.
    """
    def __init__(self):
        super().__init__()

    def forward(self, target_value, predicted_value, target_policy, predicted_policy):
        target_value = target_value.to(predicted_value.device)
        target_policy = target_policy.to(predicted_policy.device)
        value_loss = (predicted_value - target_value) ** 2
        policy_loss = torch.sum(
        return (value_loss.view(-1) + policy_loss).mean()
        return (value_loss.view(-1) + policy_loss).mean()