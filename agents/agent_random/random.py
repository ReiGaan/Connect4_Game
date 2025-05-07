import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, check_move_status, MoveStatus

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    
    valid_moves = [
        col for col in range(board.shape[1])
        if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID
    ]
    action = np.random.choice(valid_moves)
    
    return PlayerAction(action), saved_state