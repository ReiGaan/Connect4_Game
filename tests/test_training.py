import numpy as np
import torch
import os
import pytest
from collections import deque
from unittest.mock import patch, MagicMock

# =============================================
# Test Case 1: ReplayBuffer Initialization
# =============================================
def test_replay_buffer_initialization():
    buffer = ReplayBuffer(capacity=5)
    assert len(buffer) == 0
    assert buffer.buffer.maxlen == 5

# =============================================
# Test Case 2: Adding Single Experience
# =============================================
def test_replay_buffer_add_single():
    buffer = ReplayBuffer()
    exp = (np.ones((3, 6, 7)), np.array([0.5, 0.5]), 1.0)
    buffer.add(exp)
    assert len(buffer) == 1
    assert buffer.buffer[0] == exp

# =============================================
# Test Case 3: Capacity Enforcement
# =============================================
def test_replay_buffer_capacity():
    buffer = ReplayBuffer(capacity=2)
    exp1 = (np.zeros((3, 6, 7)), np.array([1.0, 0.0]), 1.0)
    exp2 = (np.ones((3, 6, 7)), np.array([0.0, 1.0]), -1.0)
    exp3 = (np.ones((3, 6, 7))*0.5, np.array([0.5, 0.5]), 0.0)
    
    buffer.add(exp1)
    buffer.add(exp2)
    buffer.add(exp3)
    
    assert len(buffer) == 2
    assert exp1 not in buffer.buffer  # Should be removed
    assert exp2 in buffer.buffer
    assert exp3 in buffer.buffer

# =============================================
# Test Case 4: Sampling from Buffer
# =============================================
def test_replay_buffer_sample():
    buffer = ReplayBuffer(capacity=3)
    experiences = [
        (np.zeros((3, 6, 7)), np.array([1.0, 0.0]), 1.0),
        (np.ones((3, 6, 7)), np.array([0.0, 1.0]), -1.0),
        (np.ones((3, 6, 7))*0.5, np.array([0.5, 0.5]), 0.0)
    ]
    
    for exp in experiences:
        buffer.add(exp)
    
    samples = buffer.sample(2)
    assert len(samples) == 2
    assert all(s in experiences for s in samples)

# =============================================
# Test Case 5: Save and Load ReplayBuffer
# =============================================
def test_replay_buffer_save_load(tmp_path):
    buffer = ReplayBuffer()
    exp = (np.eye(3)[:, :, None] * np.ones((1, 6, 7)), 
           np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]), 
           0.5)
    buffer.add(exp)
    
    file_path = tmp_path / "buffer.pt"
    buffer.save(file_path)
    
    loaded_buffer = ReplayBuffer.load(file_path)
    
    assert len(loaded_buffer) == 1
    loaded_exp = loaded_buffer.buffer[0]
    
    # Check state
    assert np.array_equal(loaded_exp[0], exp[0])
    # Check policy
    assert np.array_equal(loaded_exp[1], exp[1])
    # Check value
    assert loaded_exp[2] == exp[2]

# =============================================
# Test Case 6: Empty BoardDataset
# =============================================
def test_board_dataset_empty():
    dataset = BoardDataset([])
    assert len(dataset) == 0

# =============================================
# Test Case 7: BoardDataset Length
# =============================================
def test_board_dataset_length():
    data = [
        (np.ones((3, 6, 7)), np.array([0.5, 0.5]), 1.0),
        (np.zeros((3, 6, 7)), np.array([1.0, 0.0]), -1.0)
    ]
    dataset = BoardDataset(data)
    assert len(dataset) == 2

# =============================================
# Test Case 8: BoardDataset Item Access
# =============================================
def test_board_dataset_getitem():
    state = np.random.rand(3, 6, 7)
    policy = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    value = 0.7
    dataset = BoardDataset([(state, policy, value)])
    
    s, p, v = dataset[0]
    
    assert torch.allclose(s, torch.tensor(state, dtype=torch.float32))
    assert torch.allclose(p, torch.tensor(policy, dtype=torch.float32))
    assert torch.allclose(v, torch.tensor([value], dtype=torch.float32))

# =============================================
# Test Case 9: BoardDataset Multiple Items
# =============================================
def test_board_dataset_multiple_items():
    data = [
        (np.ones((3, 6, 7)), np.array([1.0, 0.0]), 1.0),
        (np.zeros((3, 6, 7)), np.array([0.0, 1.0]), -1.0)
    ]
    dataset = BoardDataset(data)
    
    # Check first item
    s1, p1, v1 = dataset[0]
    assert torch.allclose(s1, torch.tensor(data[0][0], dtype=torch.float32))
    
    # Check second item
    s2, p2, v2 = dataset[1]
    assert torch.allclose(p2, torch.tensor(data[1][1], dtype=torch.float32))

# =============================================
# Test Case 10: Custom Loss Calculation
# =============================================
def test_custom_loss_calculation():
    loss_fn = CustomLoss()
    
    # Test data
    values = torch.tensor([[1.0], [0.0], [-1.0]])
    pred_values = torch.tensor([[0.8], [0.2], [-0.9]])
    policies = torch.tensor([
        [0.7, 0.3],
        [0.4, 0.6],
        [0.9, 0.1]
    ])
    pred_policies = torch.tensor([
        [0.6, 0.4],
        [0.5, 0.5],
        [0.8, 0.2]
    ])
    
    loss = loss_fn(values, pred_values, policies, pred_policies)
    
    # Calculate expected loss components
    value_loss = torch.mean((values - pred_values) ** 2)
    policy_loss = -torch.mean(torch.sum(policies * torch.log(pred_policies), dim=1)
    expected_loss = value_loss + policy_loss
    
    assert torch.isclose(loss, expected_loss)

# =============================================
# Test Case 11: Custom Loss Zero Value
# =============================================
def test_custom_loss_zero_value():
    loss_fn = CustomLoss()
    
    # Test data
    values = torch.tensor([[0.0]])
    pred_values = torch.tensor([[0.0]])
    policies = torch.tensor([[0.5, 0.5]])
    pred_policies = torch.tensor([[0.5, 0.5]])
    
    loss = loss_fn(values, pred_values, policies, pred_policies)
    
    # Should be very close to zero
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

# =============================================
# Test Case 12: Model Initialization
# =============================================
def test_model_initialization():
    model = Connect4Net()
    input_tensor = torch.randn(1, 3, 6, 7)
    policy, value = model(input_tensor)
    
    assert policy.shape == (1, 7)  # 7 possible actions
    assert value.shape == (1, 1)   # Single value output

# =============================================
# Test Case 13: Model Forward Pass
# =============================================
def test_model_forward_pass():
    model = Connect4Net()
    input_tensor = torch.zeros(2, 3, 6, 7)  # Batch of 2
    policy, value = model(input_tensor)
    
    assert policy.shape == (2, 7)
    assert value.shape == (2, 1)
    
    # Policy should sum to ~1
    assert torch.allclose(torch.sum(policy, dim=1), torch.tensor([1.0, 1.0]), atol=1e-5)
    
    # Value should be between -1 and 1
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

# =============================================
# Test Case 14: Training Loop Initialization
# =============================================
def test_training_loop_initialization():
    with patch('__main__.Connect4Net') as mock_model:
        # Create a minimal config
        config = {
            'num_iterations': 1,
            'num_self_play_games': 1,
            'num_epochs': 1,
            'batch_size': 1,
            'mcts_iterations': 1,
            'learning_rate': 0.001,
            'buffer_size': 10,
            'device': 'cpu',
            'checkpoint_dir': "test_checkpoints",
            'resume_checkpoint': None
        }
        
        # Run training initialization
        trained_model = train_alphazero(**config)
        
        # Verify model was created
        mock_model.assert_called_once()

# =============================================
# Test Case 15: Checkpoint Saving
# =============================================
def test_checkpoint_saving(tmp_path):
    # Mock model and optimizer
    model = MagicMock()
    optimizer = MagicMock()
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': 5
    }
    
    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Verify file exists
    assert os.path.exists(checkpoint_path)

# =============================================
# Test Case 16: Checkpoint Loading
# =============================================
def test_checkpoint_loading(tmp_path):
    # Create a simple checkpoint
    checkpoint = {
        'model_state_dict': {'weight': torch.tensor([1.0])},
        'optimizer_state_dict': {'param': torch.tensor([2.0])},
        'iteration': 10
    }
    
    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Load checkpoint
    loaded = torch.load(checkpoint_path)
    
    # Verify content
    assert torch.equal(loaded['model_state_dict']['weight'], torch.tensor([1.0]))
    assert torch.equal(loaded['optimizer_state_dict']['param'], torch.tensor([2.0]))
    assert loaded['iteration'] == 10

# =============================================
# Test Case 17: Device Selection (CPU)
# =============================================
def test_device_selection_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4Net().to(device)
    
    # Create input tensor on correct device
    input_tensor = torch.randn(1, 3, 6, 7).to(device)
    policy, value = model(input_tensor)
    
    assert policy.device == device
    assert value.device == device

# =============================================
# Test Case 18: Board State Representation
# =============================================
def test_board_state_representation():
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0],
        [0, 2, 2, 1, 1, 0, 0]
    ])
    
    # Player 1 perspective
    player1_planes = np.stack([
        (board == 1).astype(np.float32),
        (board == 2).astype(np.float32),
        np.ones_like(board, dtype=np.float32)
    ])
    
    # Player 2 perspective
    player2_planes = np.stack([
        (board == 2).astype(np.float32),
        (board == 1).astype(np.float32),
        np.ones_like(board, dtype=np.float32)
    ])
    
    # Verify player 1 representation
    assert player1_planes[0, 3, 3] == 1  # Player 1 piece
    assert player1_planes[1, 4, 2] == 1  # Player 2 piece
    assert player1_planes[2].all() == 1   # Constant plane
    
    # Verify player 2 representation
    assert player2_planes[0, 4, 2] == 1  # Player 2 piece (now player 1 in their perspective)
    assert player2_planes[1, 3, 3] == 1  # Player 1 piece (now opponent)

# =============================================
# Test Case 19: Policy Normalization
# =============================================
def test_policy_normalization():
    visits = [10, 20, 5, 15]
    total = sum(visits)
    policy = np.array(visits) / total
    
    assert np.isclose(sum(policy), 1.0)
    assert np.allclose(policy, [0.2, 0.4, 0.1, 0.3])

# =============================================
# Test Case 20: Temperature Adjustment
# =============================================
def test_temperature_adjustment():
    policy = np.array([0.4, 0.3, 0.2, 0.1])
    temperature = 0.5
    
    adjusted = np.power(policy, 1/temperature)
    adjusted /= adjusted.sum()
    
    # Higher temperature should make distribution more uniform
    # Lower temperature should make it more peaked
    assert adjusted[0] > policy[0]  # Highest probability increases
    assert adjusted[3] < policy[3]  # Lowest probability decreases

# =============================================
# Test Case 21: Value Assignment (Win)
# =============================================
def test_value_assignment_win():
    # Create a mock game history
    game_history = [
        (None, None, 1),  # Player 1 move
        (None, None, 2),  # Player 2 move
        (None, None, 1)   # Player 1 move (wins)
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
    
    assert training_data[0][2] == 1.0  # Player 1 perspective
    assert training_data[1][2] == -1.0  # Player 2 perspective
    assert training_data[2][2] == 1.0   # Player 1 perspective

# =============================================
# Test Case 22: Value Assignment (Draw)
# =============================================
def test_value_assignment_draw():
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

# =============================================
# Test Case 23: DataLoader Integration
# =============================================
def test_dataloader_integration():
    # Create test data
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
        break  # Only test first batch