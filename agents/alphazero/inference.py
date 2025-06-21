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
    
    # Convert state to tensor (shape [1, 3, 6, 7])
    input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        policy, value = model(input_tensor)
    
    # Convert output
    policy = policy.squeeze(0).cpu().numpy()  # shape (7,)
    value = value.item()                      # scalar

    # Filter for legal actions
    legal_moves = {}
    for col in range(BOARD_COLS):
        if check_move_status(state, PlayerAction(col)) == MoveStatus.IS_VALID:
            legal_moves[col] = policy[col]

    # Normalize the policy distribution over legal moves
    total_prob = sum(legal_moves.values()) + 1e-8
    for move in legal_moves:
        legal_moves[move] /= total_prob

    return legal_moves, value
