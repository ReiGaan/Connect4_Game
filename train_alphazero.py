import os
import glob
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

from game_utils import (
    initialize_game_state,
    PLAYER1, PLAYER2,
    apply_player_action,
    check_end_state, GameState,
    get_opponent
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
    """
    Custom PyTorch dataset for training the AlphaZero model with board states.

    Args:
        data (list): List of (state, policy, value) tuples.
    """
    def __init__(self, data):
        # data: list of (state, policy, value)
        states, policies, values = zip(*data)
        self.states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        self.policies = torch.stack([torch.tensor(p, dtype=torch.float32) for p in policies])
        self.values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

    
def self_play(model_state_dict, mcts_iterations, device='cpu'):
    # reload model in each process
    device = torch.device(device)
    model = Connect4Net().to(device)
    model.load_state_dict(model_state_dict)
    agent = AlphazeroMCTSAgent(
        lambda s, p: policy_value(s, model, p, device),
        iterationnumber=mcts_iterations
    )

    board = initialize_game_state()
    player = PLAYER1
    saved = None
    history = []

    while True:
        action, saved = agent.mcts_move(board.copy(), player, saved, "SelfPlay")
        total = sum(c.visits for c in saved.children.values())
        policy = np.zeros(7, dtype=np.float32)
        for a, c in saved.children.items():
            policy[a] = c.visits / total

        # state encoding: [3,6,7]
        state = np.stack([
            (board == player).astype(np.float32),
            (board == get_opponent(player)).astype(np.float32),
            np.ones_like(board, dtype=np.float32)
        ])

        history.append((state, policy, player))
        apply_player_action(board, action, player)
        end = check_end_state(board, player)
        if end != GameState.STILL_PLAYING:
            winner = player if end == GameState.IS_WIN else None
            break
        player = get_opponent(player)

    # build training data + horizontal flips
    data = []
    for s, pol, p in history:
        val = 0.0 if winner is None else (1.0 if winner == p else -1.0)
        data.append((s, pol, val))
        # flip
        sf = np.flip(s, axis=2).copy()
        pf = np.flip(pol).copy()
        data.append((sf, pf, val))

    return data


def _self_play_job(model_state_dict, mcts_iterations, device):
    return self_play(model_state_dict, mcts_iterations, device)
    
def train_alphazero(
    num_iterations=100,
    num_self_play_games=100,
    num_epochs=10,
    batch_size=128,
    mcts_iterations=100,
    learning_rate=1e-3,
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

    # ── AUTO-PICK latest checkpoint if none specified ──
    if resume_checkpoint is None:
        pattern = os.path.join(checkpoint_dir, "iteration_*.pt")
        files = glob.glob(pattern)
        if files:
            iters = []
            for fp in files:
                name = os.path.basename(fp)
                m = re.search(r"iteration_(\d+)\.pt$", name)
                if m:
                    iters.append(int(m.group(1)))
            if iters:
                latest = max(iters)
                resume_checkpoint = f"iteration_{latest}.pt"
                print(f"[resume] auto-detected checkpoint: {resume_checkpoint}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)

    model = Connect4Net(num_residual_layers=4).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.8
    )
    loss_fn = CustomLoss()
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    start_iteration = 0

    # Resume logic
    if resume_checkpoint:
        # print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint_path = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint_path['model_state_dict'])
            optimizer.load_state_dict(checkpoint_path['optimizer_state_dict'])
            start_iteration = checkpoint_path['iteration'] + 1
            buffer_path = os.path.join(
                checkpoint_dir, f"buffer_{resume_checkpoint.split('_')[-1]}"
            )
            if os.path.exists(buffer_path):
                replay_buffer = ReplayBuffer.load(buffer_path, capacity=buffer_size)
                print(f"Resumed from iter {start_iteration}, buffer size {len(replay_buffer)}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")

    for iteration in range(start_iteration, num_iterations):
        start_time = time.time()
        print(f"\n===  Iteration {iteration+1}/{num_iterations} ===  LR={optimizer.param_groups[0]['lr']:.4f}")

        # 1) Self-play (parallel)
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        experiences = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_self_play_job, state_dict_cpu, mcts_iterations, str(device))
                for _ in range(num_self_play_games)
            ]
            for fut in as_completed(futures):
                experiences.extend(fut.result())

        print(f"Generated {len(experiences)} self-play samples")
        for exp in experiences:
            replay_buffer.add(exp)

        # 2) Training epochs
        if len(replay_buffer) >= batch_size:
            dataset = BoardDataset(replay_buffer.sample(min(len(replay_buffer), buffer_size)))
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            for epoch in range(1, num_epochs+1):
                epoch_loss = 0.0
                for states, policies, values in loader:
                    states, policies, values = states.to(device), policies.to(device), values.to(device)
                    pred_p, pred_v = model(states)
                    loss = loss_fn(values, pred_v, policies, pred_p)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                avg = epoch_loss / len(loader)
                print(f"  Epoch {epoch}/{num_epochs} — Loss: {avg:.4f}")

        scheduler.step()

        # 3) Checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"iteration_{iteration+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }, checkpoint_path)
        buffer_path = os.path.join(checkpoint_dir, f"buffer_iteration_{iteration+1}.pt")
        replay_buffer.save(buffer_path)
        print(f"Saved model + buffer in {time.time()-start_time:.1f}s to {checkpoint_dir}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Connect4 AlphaZero")
    parser.add_argument('--iterations', type=int,       default=100, help='Training iterations')
    parser.add_argument('--self_play_games', type=int,  default=100, help='Games per iteration')
    parser.add_argument('--epochs', type=int,           default=5,   help='Epochs per iteration')
    parser.add_argument('--batch_size', type=int,       default=128, help='Batch size')
    parser.add_argument('--mcts_iters', type=int,       default=100, help='MCTS simulations')
    parser.add_argument('--lr', type=float,             default=1e-3,help='Learning rate')
    parser.add_argument('--buffer_size', type=int,      default=20000,help='Replay buffer capacity')
    parser.add_argument('--device', type=str,           default='cpu',help='cpu or cuda')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints',help='Where to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume training from (e.g., iteration_10.pt)')
    args = parser.parse_args()

    config = {
        'num_iterations':       args.iterations,
        'num_self_play_games':  args.self_play_games,
        'num_epochs':           args.epochs,
        'batch_size':           args.batch_size,
        'mcts_iterations':      args.mcts_iters,
        'learning_rate':        args.lr,
        'buffer_size':          args.buffer_size,
        'device':               args.device,
        'checkpoint_dir':       args.checkpoint_dir,
        'resume_checkpoint':    args.resume
    }

    trained_model = train_alphazero(**config)
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(trained_model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")