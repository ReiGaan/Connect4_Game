import torch
import numpy as np
from game_utils import check_move_status, MoveStatus, PlayerAction, BOARD_ROWS, BOARD_COLS

def policy_value(state: np.ndarray, model: torch.nn.Module, device='cpu'):
    """
    Given a board state, returns:
    - A dict mapping legal moves (actions) to prior probabilities
    - A scalar value estimate for the current player
    """
    model.eval()
    
    current_player = 1  # Or get this from your state/game logic
    opponent_player = 2 if current_player == 1 else 1

    # Create planes
    plane_current = (state == current_player).astype(np.float32)
    plane_opponent = (state == opponent_player).astype(np.float32)
    plane_ones = np.ones_like(plane_current, dtype=np.float32)  # or zeros, if your network needs

    # Stack planes into [3, 6, 7]
    input_planes = np.stack([plane_current, plane_opponent, plane_ones])

    # Convert to tensor and add batch dim
    input_tensor = torch.tensor(input_planes, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy, value = model(input_tensor)

    policy = policy.squeeze(0).cpu().numpy()
    value = value.item()

    # Extract legal moves
    legal_moves = {}
    for col in range(7):  # BOARD_COLS
        if check_move_status(state, PlayerAction(col)) == MoveStatus.IS_VALID:
            legal_moves[col] = policy[col]

    # Normalize policy over legal moves
    total_prob = sum(legal_moves.values()) + 1e-8
    for move in legal_moves:
        legal_moves[move] /= total_prob

    return legal_moves, value
