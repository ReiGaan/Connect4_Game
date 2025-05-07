import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, check_move_status, MoveStatus

def mcts_move(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column with MCTS and return it as `action`
    return action, saved_state