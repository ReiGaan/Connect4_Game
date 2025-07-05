import numpy as np
import torch
import os
import pytest
from collections import deque
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock
from train_alphazero import ReplayBuffer, BoardDataset, CustomLoss, Connect4Net, train_alphazero

def test_replay_buffer_initialization():
    """
    Test that ReplayBuffer is correctly initialized with specified capacity.
    """
    buffer = ReplayBuffer(capacity=5)
    assert len(buffer) == 0
    assert buffer.buffer.maxlen == 5

def test_replay_buffer_add_single():
    """
    Test adding a single experience to the ReplayBuffer.
    """
    buffer = ReplayBuffer()
    exp = (np.ones((3, 6, 7)), np.array([0.5, 0.5]), 1.0)
    buffer.add(exp)
    assert len(buffer) == 1
    assert buffer.buffer[0] == exp

def test_board_dataset_empty():
    """
    Test that BoardDataset correctly handles an empty input.
    """
    dataset = BoardDataset([])
    assert len(dataset) == 0

def test_board_dataset_length():
    """
    Test that BoardDataset returns the correct length of the dataset.
    """
    data = [
        (np.ones((3, 6, 7)), np.array([0.5, 0.5]), 1.0),
        (np.zeros((3, 6, 7)), np.array([1.0, 0.0]), -1.0)
    ]
    dataset = BoardDataset(data)
    assert len(dataset) == 2

def test_board_dataset_getitem():
    """
    Test that items retrieved from BoardDataset are correctly converted to tensors.
    """
    state = np.random.rand(3, 6, 7)
    policy = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    value = 0.7
    dataset = BoardDataset([(state, policy, value)])
    
    s, p, v = dataset[0]
    
    assert torch.allclose(s, torch.tensor(state, dtype=torch.float32))
    assert torch.allclose(p, torch.tensor(policy, dtype=torch.float32))
    assert torch.allclose(v, torch.tensor([value], dtype=torch.float32))

def test_board_dataset_multiple_items():
    """
    Test that multiple items in BoardDataset are individually accessible and correct.
    """
    data = [
        (np.ones((3, 6, 7)), np.array([1.0, 0.0]), 1.0),
        (np.zeros((3, 6, 7)), np.array([0.0, 1.0]), -1.0)
    ]
    dataset = BoardDataset(data)
    
    s1, p1, v1 = dataset[0]
    assert torch.allclose(s1, torch.tensor(data[0][0], dtype=torch.float32))
    
    s2, p2, v2 = dataset[1]
    assert torch.allclose(p2, torch.tensor(data[1][1], dtype=torch.float32))

def test_model_initialization():
    """
    Test that the Connect4Net model initializes and outputs tensors of expected shape.
    """
    model = Connect4Net()
    input_tensor = torch.randn(1, 3, 6, 7)
    policy, value = model(input_tensor)
    
    assert policy.shape == (1, 7)
    assert value.shape == (1, 1)

def test_model_forward_pass():
    """
    Test that the model forward pass works for a batch of inputs, and outputs are normalized.
    """
    model = Connect4Net()
    input_tensor = torch.zeros(2, 3, 6, 7)
    policy, value = model(input_tensor)
    
    assert policy.shape == (2, 7)
    assert value.shape == (2, 1)
    assert torch.allclose(torch.sum(policy, dim=1), torch.tensor([1.0, 1.0]), atol=1e-5)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

def test_checkpoint_loading(tmp_path):
    """
    Test saving and loading a model checkpoint.
    """
    checkpoint = {
        'model_state_dict': {'weight': torch.tensor([1.0])},
        'optimizer_state_dict': {'param': torch.tensor([2.0])},
        'iteration': 10
    }
    
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path)
    
    assert torch.equal(loaded['model_state_dict']['weight'], torch.tensor([1.0]))
    assert torch.equal(loaded['optimizer_state_dict']['param'], torch.tensor([2.0]))
    assert loaded['iteration'] == 10

def test_board_state_representation():
    """
    Test that the board state is represented correctly for both players.
    """
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0],
        [0, 2, 2, 1, 1, 0, 0]
    ])
    
    player1_planes = np.stack([
        (board == 1).astype(np.float32),
        (board == 2).astype(np.float32),
        np.ones_like(board, dtype=np.float32)
    ])
    
    player2_planes = np.stack([
        (board == 2).astype(np.float32),
        (board == 1).astype(np.float32),
        np.ones_like(board, dtype=np.float32)
    ])
    
    assert player1_planes[0, 3, 3] == 1
    assert player1_planes[1, 4, 2] == 1
    assert player1_planes[2].all() == 1

    assert player2_planes[0, 4, 2] == 1
    assert player2_planes[1, 3, 3] == 1

def test_policy_normalization():
    """
    Test that the policy vector is correctly normalized to sum to 1.
    """
    visits = [10, 20, 5, 15]
    total = sum(visits)
    policy = np.array(visits) / total
    
    assert np.isclose(sum(policy), 1.0)
    assert np.allclose(policy, [0.2, 0.4, 0.1, 0.3])

def test_temperature_adjustment():
    """
    Test temperature scaling of a policy distribution.
    """
    policy = np.array([0.4, 0.3, 0.2, 0.1])
    temperature = 0.5
    adjusted = np.power(policy, 1/temperature)
    adjusted /= adjusted.sum()
    
    assert adjusted[0] > policy[0]
    assert adjusted[3] < policy[3]

def test_value_assignment_win():
    """
    Test correct value assignment for a game ending in a win.
    """
    game_history = [
        (None, None, 1),
        (None, None, 2),
        (None, None, 1)
    ]
    winner = 1
    
    training_data = []
    for state_rep, policy, player in game_history:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_data.append((state_rep, policy, value))
    
    assert training_data[0][2] == 1.0
    assert training_data[1][2] == -1.0
    assert training_data[2][2] == 1.0

def test_value_assignment_draw():
    """
    Test correct value assignment for a game ending in a draw.
    """
    game_history = [
        (None, None, 1),
        (None, None, 2),
        (None, None, 1)
    ]
    winner = None
    
    training_data = []
    for state_rep, policy, player in game_history:
        value = 0.0 if winner is None else (1.0 if winner == player else -1.0)
        training_data.append((state_rep, policy, value))
    
    assert all(item[2] == 0.0 for item in training_data)

def test_dataloader_integration():
    """
    Test integration of BoardDataset with PyTorch DataLoader.
    """
    data = [
        (np.random.rand(3, 6, 7), np.random.rand(7), 1.0),
        (np.random.rand(3, 6, 7), np.random.rand(7), -1.0)
    ]
    dataset = BoardDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for states, policies, values in loader:
        assert states.shape == (2, 3, 6, 7)
        assert policies.shape == (2, 7)
        assert values.shape == (2, 1)
        break
