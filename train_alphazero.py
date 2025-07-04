import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
from collections import deque
from game_utils import initialize_game_state, BoardPiece, PLAYER1, PLAYER2, apply_player_action, check_end_state, GameState, get_opponent, PlayerAction
from agents.alphazero.network import Connect4Net, CustomLoss
from agents.alphazero.inference import policy_value
from agents.agent_MCTS.alphazero_mcts import AlphazeroMCTSAgent

# =============================
# 1. Data Collection Components
# =============================
class ReplayBuffer:
    """Stores game experiences for training"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience):
        """Add a game experience to the buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class BoardDataset(Dataset):
    """PyTorch Dataset for training"""
    def __init__(self, data):
        self.states = []
        self.policies = []
        self.values = []
        
        for state, policy, value in data:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
            
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor([self.values[idx]], dtype=torch.float32)
        return state, policy, value

# =============================
# 2. Self-Play Implementation
# =============================
def self_play(model, device, mcts_iterations=100, temperature=1.0):
    """
    Play a game using MCTS with the current model to generate training data
    
    Args:
        model: Current neural network
        device: Torch device (CPU/GPU)
        mcts_iterations: Number of MCTS simulations per move
        temperature: Controls exploration (1.0 = more exploration, 0.0 = greedy)
    
    Returns:
        List of (state, policy, value) tuples for training
    """
    # Initialize game
    board = initialize_game_state()
    agent = AlphazeroMCTSAgent(
        lambda state: policy_value(state, model, device),
        iterationnumber=mcts_iterations
    )
    saved_state = None
    current_player = PLAYER1
    game_history = []
    
    while True:
        # Get MCTS policy
        action, saved_state = agent.mcts_move(
            board.copy(), current_player, saved_state, "SelfPlay"
        )
        
        # Get visit counts and normalize to policy
        total_visits = sum(child.visits for child in saved_state.children.values())
        policy = np.zeros(7)  # 7 columns
        
        for a, child in saved_state.children.items():
            policy[a] = child.visits / total_visits
        
        # Apply temperature
        if temperature != 1.0:
            policy = np.power(policy, 1/temperature)
            policy /= np.sum(policy)
        
        # Create state representation for current player
        state_rep = np.stack([
            (board == current_player).astype(np.float32),
            (board == get_opponent(current_player)).astype(np.float32),
            np.ones_like(board, dtype=np.float32)
        ])
        
        # Save experience
        game_history.append((state_rep, policy, current_player))
        
        # Apply action
        apply_player_action(board, action, current_player)
        
        # Check game end
        end_state = check_end_state(board, current_player)
        if end_state != GameState.STILL_PLAYING:
            # Determine winner
            if end_state == GameState.IS_WIN:
                winner = current_player
            else:
                winner = None
            break
                
        current_player = get_opponent(current_player)
    
    # Assign values to all states in the game
    training_data = []
    for i, (state_rep, policy, player) in enumerate(game_history):
        # Value from perspective of player at that state
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
            
        training_data.append((state_rep, policy, value))
    
    return training_data

# =============================
# 3. Training Loop
# =============================
def train_alphazero(
    num_iterations=100, 
    num_self_play_games=100,
    num_epochs=10,
    batch_size=128,
    mcts_iterations=100,
    learning_rate=0.001,
    buffer_size=10000,
    device='cpu'
):
    """Main training loop for AlphaZero"""
    # Initialize model
    model = Connect4Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = CustomLoss()
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Create directory for checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration+1}/{num_iterations} ===")
        start_time = time.time()
        
        # Self-play phase
        print(f"Playing {num_self_play_games} self-play games...")
        for game_idx in range(num_self_play_games):
            game_data = self_play(model, device, mcts_iterations)
            for experience in game_data:
                replay_buffer.add(experience)
            
            if (game_idx + 1) % 10 == 0:
                print(f"  Completed {game_idx+1}/{num_self_play_games} games")
        
        # Train on collected data
        print(f"Training on {len(replay_buffer)} experiences...")
        if len(replay_buffer) > batch_size:
            # Sample training data
            train_data = replay_buffer.sample(min(len(replay_buffer), 2048))
            train_dataset = BoardDataset(train_data)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Training epoch
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for states, policies, values in train_loader:
                    states = states.to(device)
                    policies = policies.to(device)
                    values = values.to(device)
                    
                    # Forward pass
                    pred_policies, pred_values = model(states)
                    
                    # Compute loss
                    loss = loss_fn(values, pred_values, policies, pred_policies)
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/iteration_{iteration+1}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Print iteration stats
        iteration_time = time.time() - start_time
        print(f"Iteration completed in {iteration_time:.2f} seconds")
    
    return model

# =============================
# 4. Main Execution
# =============================
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    config = {
        'num_iterations': 100,
        'num_self_play_games': 100,
        'num_epochs': 10,
        'batch_size': 128,
        'mcts_iterations': 100,
        'learning_rate': 0.001,
        'buffer_size': 10000,
        'device': device
    }
    
    # Start training
    trained_model = train_alphazero(**config)
    
    # Save final model
    torch.save(trained_model.state_dict(), "alphazero_final_model.pt")
    print("Training complete! Saved final model as alphazero_final_model.pt")