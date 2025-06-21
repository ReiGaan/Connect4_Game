import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, check_move_status, MoveStatus
from metrics.metrics import GameMetrics
def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: SavedState | None, player_name: str, metrics: GameMetrics | None = None,
) -> tuple[PlayerAction, SavedState | None]:
    """
    Generates a random valid move for the current player.
    
    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The current player's piece.
        saved_state (SavedState | None): The saved state from the previous move, or None.
    Returns:
        tuple[PlayerAction, SavedState | None]: A tuple containing the randomly selected
        valid move as a PlayerAction and the (possibly unchanged) saved state.
    """
    valid_moves = [
        col for col in range(board.shape[1])
        if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID
    ]
    action = np.random.choice(valid_moves)
    
    return PlayerAction(action), saved_state