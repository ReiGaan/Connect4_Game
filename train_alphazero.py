import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import time
from collections import deque
from game_utils import (
    initialize_game_state, BoardPiece, PLAYER1, PLAYER2,
    apply_player_action, check_end_state, GameState,
    get_opponent, PlayerAction
)
from agents.alphazero.network import Connect4Net, CustomLoss
from agents.alphazero.inference import policy_value
from agents.agent_MCTS.alphazero_mcts import AlphazeroMCTSAgent


class ReplayBuffer:
    """
    A buffer to store gameplay experiences for training the AlphaZero model.

    Args:
        capacity (int): Maximum number of experiences to store.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """
        Add a game experience to the buffer.

        Args:
            experience (tuple): (state, policy, value) tuple.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            List of sampled experiences.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        """
        Save the buffer to a file.

        Args:
            path (str): File path to save the buffer.
        """
        torch.save(list(self.buffer), path)

    @staticmethod
    def load(path, capacity=10000):
        """
        Load a buffer from a file.

        Args:
            path (str): File path to load from.
            capacity (int): Maximum buffer capacity.

        Returns:
            ReplayBuffer: Loaded replay buffer.
        """
        buffer = ReplayBuffer(capacity)
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            buffer.buffer = deque(data, maxlen=capacity)
        return buffer


class BoardDataset(Dataset):
    """
    Custom PyTorch dataset for training the AlphaZero model with board states.

    Args:
        data (list): List of (state, policy, value) tuples.
    """
    def __init__(self, data):
        """Store data as tensors for faster access during training."""
        if len(data) == 0:
            # Create empty tensors when no data is provided
            self.states = torch.empty((0,), dtype=torch.float32)
            self.policies = torch.empty((0,), dtype=torch.float32)
            self.values = torch.empty((0, 1), dtype=torch.float32)
            return

        states_list, policies_list, values_list = zip(*data)
        self.states = torch.stack(
            [torch.tensor(s, dtype=torch.float32) for s in states_list]
        )
        self.policies = torch.stack(
            [torch.tensor(p, dtype=torch.float32) for p in policies_list]
        )
        self.values = torch.tensor(values_list, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """
        Retrieve a single training sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (state, policy, value) tensors.
        """
        state = self.states[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        return state, policy, value


def self_play(model, device, mcts_iterations=100, temperature=1.0):
    """
    Play a single self-play game using MCTS to generate training data.

    Args:
        model (torch.nn.Module): The neural network model.
        device (str): Device to run inference on ('cpu' or 'cuda').
        mcts_iterations (int): Number of MCTS simulations per move.
        temperature (float): Exploration temperature.

    Returns:
        list: List of (state, policy, value) tuples.
    """
    board = initialize_game_state()
    agent = AlphazeroMCTSAgent(
        lambda state, player: policy_value(state, model, player, device),
        iterationnumber=mcts_iterations
    )
    saved_state = None
    current_player = PLAYER1
    game_history = []

    while True:
        action, saved_state = agent.mcts_move(
            board.copy(), current_player, saved_state, "SelfPlay"
        )

        total_visits = sum(child.visits for child in saved_state.children.values())
        policy = np.zeros(7)
        for a, child in saved_state.children.items():
            policy[a] = child.visits / total_visits

        if temperature != 1.0:
            policy = np.power(policy, 1 / temperature)
            policy /= np.sum(policy)

        state_rep = np.stack([
            (board == current_player).astype(np.float32),
            (board == get_opponent(current_player)).astype(np.float32),
            np.ones_like(board, dtype=np.float32)
        ])

        game_history.append((state_rep, policy, current_player))

        apply_player_action(board, action, current_player)

        end_state = check_end_state(board, current_player)
        if end_state != GameState.STILL_PLAYING:
            winner = current_player if end_state == GameState.IS_WIN else None
            break

        current_player = get_opponent(current_player)

    training_data = []
    for state_rep, policy, player in game_history:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_data.append((state_rep, policy, value))

    return training_data


def train_alphazero(
    num_iterations=100,
    num_self_play_games=100,
    num_epochs=10,
    batch_size=128,
    mcts_iterations=100,
    learning_rate=0.001,
    buffer_size=10000,
    device='cpu',
    checkpoint_dir="checkpoints",
    resume_checkpoint=None
):
    """
    Main training loop for AlphaZero including checkpointing and self-play.

    Args:
        num_iterations (int): Number of training iterations.
        num_self_play_games (int): Games to generate per iteration.
        num_epochs (int): Epochs per training step.
        batch_size (int): Mini-batch size.
        mcts_iterations (int): MCTS simulations per move.
        learning_rate (float): Learning rate for optimizer.
        buffer_size (int): Capacity of the replay buffer.
        device (str): Computation device ('cpu' or 'cuda').
        checkpoint_dir (str): Path to save checkpoints.
        resume_checkpoint (str or None): Resume from this checkpoint if provided.

    Returns:
        model (torch.nn.Module): Trained model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = Connect4Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = CustomLoss()
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    start_iteration = 0

    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['iteration'] + 1

            buffer_path = os.path.join(
                checkpoint_dir, f"buffer_{resume_checkpoint.split('_')[-1]}"
            )
            if os.path.exists(buffer_path):
                replay_buffer = ReplayBuffer.load(buffer_path, capacity=buffer_size)
                print(f"Loaded replay buffer with {len(replay_buffer)} experiences")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")

    for iteration in range(start_iteration, num_iterations):
        print(f"\n=== Iteration {iteration+1}/{num_iterations} ===")
        start_time = time.time()

        print(f"Playing {num_self_play_games} self-play games...")
        for game_idx in range(num_self_play_games):
            game_data = self_play(model, device, mcts_iterations)
            for experience in game_data:
                replay_buffer.add(experience)

            print(f"  Completed {game_idx+1}/{num_self_play_games} games")

        print(f"Training on {len(replay_buffer)} experiences...")
        if len(replay_buffer) > batch_size:
            sample_size = min(len(replay_buffer), 2048)
            train_data = replay_buffer.sample(sample_size)
            train_dataset = BoardDataset(train_data)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for states, policies, values in train_loader:
                    states = states.to(device)
                    policies = policies.to(device)
                    values = values.to(device)

                    pred_policies, pred_values = model(states)
                    loss = loss_fn(values, pred_values, policies, pred_policies)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"iteration_{iteration+1}.pt")
        buffer_path = os.path.join(checkpoint_dir, f"buffer_{iteration+1}.pt")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
        }, checkpoint_path)

        replay_buffer.save(buffer_path)

        print(f"Saved checkpoint to {checkpoint_path}")
        print(f"Saved replay buffer to {buffer_path}")
        print(f"Iteration completed in {time.time() - start_time:.2f} seconds")

    return model


if __name__ == "__main__":
    """
    Entry point for training the AlphaZero model. Parses CLI arguments and starts training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        'num_iterations': 100,
        'num_self_play_games': 100,
        'num_epochs': 10,
        'batch_size': 128,
        'mcts_iterations': 100,
        'learning_rate': 0.001,
        'buffer_size': 10000,
        'device': device,
        'checkpoint_dir': "checkpoints",
        'resume_checkpoint': "iteration_2.pt"
    }

    import argparse
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume training from (e.g., iteration_10.pt)')
    args = parser.parse_args()

    if args.resume:
        config['resume_checkpoint'] = args.resume

    trained_model = train_alphazero(**config)

    final_model_path = "alphazero_final_model.pt"
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Training complete! Saved final model as {final_model_path}")
