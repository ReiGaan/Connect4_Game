import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column randomly and return it as `action`
    return action, saved_state