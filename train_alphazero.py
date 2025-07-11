import os
import glob
import re
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        # experience = (state: np.ndarray[3,6,7], policy: dict[int,float], value: float)
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
        # **always** build a length-7 vector, filling zeros for illegal moves
        self.policies = torch.stack([
            torch.tensor([pol.get(col, 0.0) for col in range(BOARD_COLS)], dtype=torch.float32)
            for pol in policies
        ])
        self.values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def self_play(model_state_dict, mcts_iterations, device='cpu'):
    device = torch.device(device)
    model = Connect4Net().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    agent = AlphazeroMCTSAgent(
        lambda s, p: policy_value(s, model, p, device),
        iterationnumber=mcts_iterations
    )

    board = initialize_game_state()
    player = PLAYER1
    saved = None
    # we’ll store (state_planes, π, mover)
    history: list[tuple[np.ndarray, dict[int,float], int]] = []
    move_number = 0

    while True:
        # --- 1) compute original priors+value at root
        orig_pv = agent.policy_value
        priors, value = orig_pv(board, player)

        # --- 2) inject Dirichlet noise
        eps, alpha = 0.25, 0.3
        moves = list(priors.keys())
        p = np.array([priors[a] for a in moves], dtype=np.float32)
        noise = np.random.dirichlet([alpha] * len(moves))
        noisy = {a: (1-eps)*p[i] + eps*noise[i] for i,a in enumerate(moves)}

        # --- 3) override pv for the first call only
        called = False
        def pv_root(s, pply):
            nonlocal called
            if not called:
                called = True
                return noisy, value
            return orig_pv(s, pply)

        agent.policy_value = pv_root

        # --- 4) run MCTS from this root
        action, saved = agent.mcts_move(board, player, saved, "SelfPlay")

        # --- 5) restore real policy_value
        agent.policy_value = orig_pv

        # --- 6) build π from visit counts & apply temperature
        visits = np.array([c.visits for c in saved.children.values()], dtype=np.float32)
        actions = list(saved.children.keys())
        π = {a: saved.children[a].visits for a in actions}
        total = visits.sum()
        for a in π:
            π[a] /= total

        if move_number < 30:
            probs = visits / total
            action = np.random.choice(actions, p=probs)
        else:
            action = max(saved.children.items(), key=lambda x: x[1].visits)[0]

        # --- 7) record raw state, π, and mover
        state_planes = np.stack([
            (board == player).astype(np.float32),
            (board == get_opponent(player)).astype(np.float32),
            np.ones_like(board, dtype=np.float32)
        ])
        history.append((state_planes, π, player))

        # --- 8) apply move
        apply_player_action(board, action, player)
        result = check_end_state(board, player)
        if result != GameState.STILL_PLAYING:
            winner = player if result == GameState.IS_WIN else None
            break

        # --- 9) next turn
        player = get_opponent(player)
        move_number += 1

    # --- 10) compute z and augment with flips
    data = []
    for state_planes, π, mover in history:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if winner == mover else -1.0
        data.append((state_planes, π, z))
        sf = np.flip(state_planes, axis=2).copy()
        pf = {6 - a: v for a, v in π.items()}
        data.append((sf, pf, z))

    return data


def _self_play_job(model_state_dict, mcts_iterations, device):
    return self_play(model_state_dict, mcts_iterations, device)


def train_alphazero(
    num_iterations=1000,
    num_self_play_games=1000,
    num_epochs=10,
    batch_size=128,
    mcts_iterations=100,
    learning_rate=1e-3,
    buffer_size=50000,
    device='cpu',
    checkpoint_dir="checkpoints",
    resume_checkpoint=None
):
    # auto-pick latest checkpoint if none specified
    if resume_checkpoint is None:
        pattern = os.path.join(checkpoint_dir, "iteration_*.pt")
        files = glob.glob(pattern)
        iters = [int(re.search(r"iteration_(\d+)\.pt$", os.path.basename(f)).group(1))
                 for f in files if re.search(r"iteration_(\d+)\.pt$", f)]
        if iters:
            latest = max(iters)
            resume_checkpoint = f"iteration_{latest}.pt"
            print(f"[resume] auto-detected checkpoint: {resume_checkpoint}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)

    model = Connect4Net(num_residual_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    loss_fn = CustomLoss()
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    start_iteration = 0

    # resume logic
    if resume_checkpoint:
        cp = os.path.join(checkpoint_dir, resume_checkpoint)
        if os.path.exists(cp):
            ck = torch.load(cp, map_location=device)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_iteration = ck['iteration'] + 1
            buf_path = os.path.join(
                checkpoint_dir, f"buffer_iteration{resume_checkpoint.split('_')[-1]}"
            )
            if os.path.exists(buf_path):
                replay_buffer = ReplayBuffer.load(buf_path, capacity=buffer_size)
                print(f"Resumed at iter {start_iteration}, buffer size {len(replay_buffer)}")

    for iteration in range(start_iteration, num_iterations):
        t0 = time.time()
        print(f"\n=== Iteration {iteration+1}/{num_iterations} — LR={optimizer.param_groups[0]['lr']:.4f} ===")

        # 1) self-play
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

        # 2) training
        if len(replay_buffer) >= batch_size:
            dataset = BoardDataset(replay_buffer.sample(min(len(replay_buffer), buffer_size)))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            for epoch in range(1, num_epochs+1):
                loss_accum = 0.0
                for states, policies, values in loader:
                    states, policies, values = states.to(device), policies.to(device), values.to(device)
                    pred_p, pred_v = model(states)
                    loss = loss_fn(values, pred_v, policies, pred_p)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    loss_accum += loss.item()
                print(f"  Epoch {epoch}/{num_epochs} — Loss: {loss_accum/len(loader):.4f}")

        scheduler.step()

        # 3) checkpoint
        cp_path = os.path.join(checkpoint_dir, f"iteration_{iteration+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }, cp_path)
        buf_path = os.path.join(checkpoint_dir, f"buffer_{iteration+1}.pt")
        replay_buffer.save(buf_path)
        print(f"Saved checkpoint {iteration+1} in {time.time()-t0:.1f}s")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Connect4 AlphaZero")
    parser.add_argument('--iterations', type=int,       default=1000)
    parser.add_argument('--self_play_games', type=int,  default=1000)
    parser.add_argument('--epochs', type=int,           default=10)
    parser.add_argument('--batch_size', type=int,       default=128)
    parser.add_argument('--mcts_iters', type=int,       default=100)
    parser.add_argument('--lr', type=float,             default=1e-3)
    parser.add_argument('--buffer_size', type=int,      default=50000)
    parser.add_argument('--device', type=str,           default='cpu')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    parser.add_argument('--resume', type=str,           default=None)
    args = parser.parse_args()

    cfg = {
        'num_iterations':      args.iterations,
        'num_self_play_games': args.self_play_games,
        'num_epochs':          args.epochs,
        'batch_size':          args.batch_size,
        'mcts_iterations':     args.mcts_iters,
        'learning_rate':       args.lr,
        'buffer_size':         args.buffer_size,
        'device':              args.device,
        'checkpoint_dir':      args.checkpoint_dir,
        'resume_checkpoint':   args.resume
    }

    trained_model = train_alphazero(**cfg)
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(trained_model.state_dict(), final_path)
    print(f"\nTraining complete — final model at {final_path}")
