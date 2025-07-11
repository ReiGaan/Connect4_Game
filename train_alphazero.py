#!/usr/bin/env python3
import os
import glob
import re
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque

from game_utils import (
    initialize_game_state,
    PLAYER1,
    apply_player_action,
    check_end_state,
    GameState,
    get_opponent,
    BOARD_COLS,
)
from agents.alphazero.network import Connect4Net, CustomLoss
from agents.alphazero.inference import policy_value
from agents.agent_MCTS.alphazero_mcts import AlphazeroMCTSAgent


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        # experience = (state: np.ndarray[3,6,7], policy: np.ndarray[7], value: float)
        self.buffer.append(experience)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        torch.save(list(self.buffer), path)

    @staticmethod
    def load(path, capacity=10000):
        buf = ReplayBuffer(capacity)
        if os.path.exists(path):
            data = torch.load(path)
            buf.buffer = deque(data, maxlen=capacity)
        return buf


class BoardDataset(Dataset):
    def __init__(self, data):
        states, policies, values = zip(*data)
        self.states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        self.policies = torch.stack([torch.tensor(p, dtype=torch.float32) for p in policies])
        self.values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def self_play(model: torch.nn.Module, device: torch.device, mcts_iterations: int):
    """
    Play one self-play game, returning a list of (state, policy, value) examples with flips.
    """
    model.eval()
    # Wrap the inference function to match AlphazeroMCTSAgent's expected signature
    def wrapped_pv(state, parent):
        return policy_value(state, model, parent, device)

    agent = AlphazeroMCTSAgent(wrapped_pv, iterationnumber=mcts_iterations)
    board = initialize_game_state()
    current_player = PLAYER1
    saved = None
    history = []

    # Play until terminal
    while True:
        action, next_node = agent.mcts_move(board.copy(), current_player, saved, "SelfPlay")

        # Safely handle cases where no node is returned
        root_node = next_node.parent if next_node is not None else None

        # Build policy from visit counts using the search root
        if root_node is not None and root_node.children:
            visits = np.array(
                [child.visits for child in root_node.children.values()], dtype=np.float32
            )
            actions = list(root_node.children.keys())
            policy = np.zeros(BOARD_COLS, dtype=np.float32)
            total_visits = visits.sum()
            if total_visits > 0:
                for a, v in zip(actions, visits):
                    policy[a] = v / total_visits
        else:
            # Default to a uniform policy if visit information is unavailable
            policy = np.full(BOARD_COLS, 1.0 / BOARD_COLS, dtype=np.float32)

        # Keep the returned best child for next iteration
        saved = next_node

        # Record current position
        state_planes = np.stack([
            (board == current_player).astype(np.float32),
            (board == get_opponent(current_player)).astype(np.float32),
            np.ones_like(board, dtype=np.float32),
        ])
        history.append((state_planes, policy, current_player))

        # Apply move
        apply_player_action(board, action, current_player)
        end_state = check_end_state(board, current_player)
        if end_state != GameState.STILL_PLAYING:
            winner = current_player if end_state == GameState.IS_WIN else None
            break

        current_player = get_opponent(current_player)

    # Build training examples with value z and horizontal-flip augmentation
    training_data = []
    for state_planes, policy, player in history:
        if winner is None:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0

        # original
        training_data.append((state_planes, policy, z))
        # flipped
        flipped_state = np.flip(state_planes, axis=2).copy()
        flipped_policy = np.flip(policy).copy()
        training_data.append((flipped_state, flipped_policy, z))

    return training_data


def train_alphazero(
    num_iterations: int = 100,
    num_self_play_games: int = 500,
    num_epochs: int = 10,
    batch_size: int = 128,
    mcts_iterations: int = 100,
    learning_rate: float = 1e-3,
    buffer_size: int = 30000,
    device_str: str = 'cpu',
    checkpoint_dir: str = "checkpoints",
    resume_checkpoint: str = None
):
    # Auto‐detect latest checkpoint if not specified
    if resume_checkpoint is None:
        pattern = os.path.join(checkpoint_dir, "iteration_*.pt")
        files = glob.glob(pattern)
        iters = [int(re.search(r"iteration_(\d+)\.pt$", os.path.basename(f)).group(1))
                 for f in files if re.search(r"iteration_(\d+)\.pt$", os.path.basename(f))]
        if iters:
            latest = max(iters)
            resume_checkpoint = f"iteration_{latest}.pt"
            print(f"[resume] auto-detected checkpoint: {resume_checkpoint}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device_str)

    model = Connect4Net(num_residual_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    loss_fn = CustomLoss()
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    start_iter = 0

    # Resume logic
    if resume_checkpoint:
        cp = os.path.join(checkpoint_dir, resume_checkpoint)
        if os.path.exists(cp):
            ck = torch.load(cp, map_location=device)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_iter = ck['iteration'] + 1
            buf_path = os.path.join(checkpoint_dir, f"buffer_iteration{resume_checkpoint.split('_')[-1]}.pt")
            if os.path.exists(buf_path):
                replay_buffer = ReplayBuffer.load(buf_path, capacity=buffer_size)
                print(f"[resume] buffer loaded, size={len(replay_buffer)}")

    # Main loop
    for iteration in range(start_iter, num_iterations):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\n=== Iteration {iteration+1}/{num_iterations} — LR={lr:.6f} ===")

        # 1) Self-play (synchronous)
        experiences = []
        for i in range(num_self_play_games):
            batch = self_play(model, device, mcts_iterations)
            print(f"  Game {i+1}/{num_self_play_games}: {len(batch)} samples")
            experiences.extend(batch)
        print(f"  → Total self-play samples: {len(experiences)}")

        # Add to replay buffer
        for exp in experiences:
            replay_buffer.add(exp)

        # 2) Training
        if len(replay_buffer) >= batch_size:
            dataset = BoardDataset(replay_buffer.sample(min(len(replay_buffer), buffer_size)))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            for epoch in range(1, num_epochs+1):
                loss_accum = 0.0
                for states, policies, values in loader:
                    states = states.to(device)
                    policies = policies.to(device)
                    values = values.to(device)
                    pred_p, pred_v = model(states)
                    loss = loss_fn(values, pred_v, policies, pred_p)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    loss_accum += loss.item()
                avg_loss = loss_accum / len(loader)
                print(f"    Epoch {epoch}/{num_epochs} — Loss: {avg_loss:.4f}")
        else:
            print(f"  Skipping training (buffer size {len(replay_buffer)} < batch_size {batch_size})")

        scheduler.step()

        # 3) Checkpoint
        cp_path = os.path.join(checkpoint_dir, f"iteration_{iteration+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }, cp_path)
        buf_path = os.path.join(checkpoint_dir, f"buffer_iteration{iteration+1}.pt")
        replay_buffer.save(buf_path)
        print(f"  Checkpoint saved ({time.time() - t0:.1f}s)")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Connect4 AlphaZero")
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--self_play_games', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mcts_iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--buffer_size', type=int, default=30000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = {
        'num_iterations':      args.iterations,
        'num_self_play_games': args.self_play_games,
        'num_epochs':          args.epochs,
        'batch_size':          args.batch_size,
        'mcts_iterations':     args.mcts_iters,
        'learning_rate':       args.lr,
        'buffer_size':         args.buffer_size,
        'device_str':          args.device,
        'checkpoint_dir':      args.checkpoint_dir,
        'resume_checkpoint':   args.resume
    }

    trained_model = train_alphazero(**cfg)
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(trained_model.state_dict(), final_path)
    print(f"\nTraining complete — final model at {final_path}")
