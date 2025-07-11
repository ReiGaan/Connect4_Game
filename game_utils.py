from typing import Callable, Optional, Any
from enum import Enum
import numpy as np
import scipy.signal
from .metrics.metrics import GameMetrics

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(
    1
)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(
    2
)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(" ")
PLAYER1_PRINT = BoardPiecePrint("X")
PLAYER2_PRINT = BoardPiecePrint("O")

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    """
    Enumeration representing the possible states of a Connect4 game.
    Attributes:
        IS_WIN (int): Indicates that the game has been won by a player.
        IS_DRAW (int): Indicates that the game ended in a draw.
        STILL_PLAYING (int): Indicates that the game is still ongoing.
    """
    
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class MoveStatus(Enum):
    """
    Enumeration representing the possible statuses of a move in the Connect4 game.
    Attributes:
        IS_VALID (int): Indicates the move is valid.
        WRONG_TYPE (str): Indicates the input does not have the correct type (should be PlayerAction).
        OUT_OF_BOUNDS (str): Indicates the input is out of bounds.
        FULL_COLUMN (str): Indicates the selected column is already full.
    """
    IS_VALID = 1
    WRONG_TYPE = "Input does not have the correct type (PlayerAction)."
    OUT_OF_BOUNDS = "Input is out of bounds."
    FULL_COLUMN = "Selected column is full."


class SavedState:
    """
    A class to represent the saved state of a Connect4 game.
    """
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, 'SavedState | None', str, 'GameMetrics | None'],
    tuple['PlayerAction', 'SavedState | None']
]


def initialize_game_state() -> np.ndarray:
    """
    Initializes and returns the starting state of the Connect4 game board.
    Returns:
        np.ndarray: A NumPy array of shape BOARD_SHAPE filled with NO_PLAYER values,
        representing an empty game board.
    """
    return np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Returns a string representation of the Connect4 board in a human-readable format.
    Args:
        board (np.ndarray): A 2D NumPy array representing the game board, where each cell contains a value corresponding to a player or empty space.
    Returns:
        str: A formatted string displaying the board with borders, player pieces, and column indices.
    Notes:
        - The mapping from board values to printable characters is defined by the constants NO_PLAYER_PRINT, PLAYER1_PRINT, and PLAYER2_PRINT.
    """
 
    piece_to_char = {
        NO_PLAYER: NO_PLAYER_PRINT,
        PLAYER1: PLAYER1_PRINT,
        PLAYER2: PLAYER2_PRINT,
    }

    border = "|==============|"
    column_indices = "|0 1 2 3 4 5 6 |"
    rows = []
    rows.append(border)  # Top border
    # Flip board to match that [0,0] is lower-left
    for row in board[::1]:
        line = "|"
        for cell in row:
            line += piece_to_char[cell] + " "
        line += "|"
        rows.append(line)
    rows.append(border)  # Bottom border
    rows.append(column_indices)  # Column indices

    return "\n".join(rows)


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
     Args:
        pp_board (strin): A formatted string displaying the board with borders, player pieces, and column indices.
    Returns:
        board (np.ndarray): A 2D NumPy array representing the game board, where each cell contains a value corresponding to a player or empty space.
    
    """
    char_to_piece = {
        NO_PLAYER_PRINT: NO_PLAYER,
        PLAYER1_PRINT: PLAYER1,
        PLAYER2_PRINT: PLAYER2,
    }

    pp_board = pp_board.splitlines()
    board_array = np.zeros(BOARD_SHAPE, dtype=int)

    for i, row in enumerate(pp_board[1:-2]):
        column = 0
        for j, cell in enumerate(row[1:-1]):
            if j % 2 == 0:
                board_array[i, column] = char_to_piece[cell]
                column += 1
    return board_array


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input
    board should be modified in place, such that it's not necessary to return
    something.

    Args:
        board (np.ndarray): The current game board.
        action (PlayerAction): The column to play in.
        player (BoardPiece): The player making the move.
    """
    for i, cell in enumerate(board[::-1, action]):
        if cell == NO_PLAYER:
            board[BOARD_ROWS - 1 - i, action] = player
            return
    raise ValueError(f"Column {action+1} is full.")

def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player to check for.

    Returns:
        bool: True if there are four in a row, False otherwise.
    """
    return (
        test_connect_horizontal(board, player)
        or test_connect_vertical(board, player)
        or test_connect_diagonal(board, player))
 


def test_connect_horizontal(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in horizontal. Returns False otherwise.

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player to check for.

    Returns:
        bool: True if there are four in a horizontal, False otherwise.
    """

    # Create a mask where the player's pieces are 1, others are 0
    mask = (board == player).astype(int)
    # Define a horizontal kernel of length 4
    kernel = np.ones((1, 4), dtype=int)
    # Convolve mask with kernel
    conv = scipy.signal.convolve2d(mask, kernel, mode="valid")
    # Check if any value is 4 (i.e., four in a row)
    return np.any(conv == 4)


def test_connect_vertical(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged in a
    vertical line. Returns False otherwise.

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player to check for.

    Returns:
        bool: True if there are four in a vertical, False otherwise.
    """
    for col in range(board.shape[1]):
        for row in range(board.shape[0] - 3): 
            if np.all(board[row:row+4, col] == player):
                return True
    return False

def test_connect_diagonal(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in a diagonal line. Returns False otherwise.

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player to check for.

    Returns:
        bool: True if there are four in a diagonal, False otherwise.
    """
    mask = (board == player).astype(int)
    kernel_top_left = np.eye(4, dtype=int)
    kernel_bottom_left = np.fliplr(kernel_top_left)
    conv_main = scipy.signal.convolve2d(mask, kernel_top_left, mode="valid")
    conv_anti = scipy.signal.convolve2d(mask, kernel_bottom_left, mode="valid")
    return np.any(conv_main == 4) or np.any(conv_anti == 4)


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)? 

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player whose move was last played.

    Returns:
        GameState: The current state of the game (win, draw, or still playing).
    """
    opponent = get_opponent(player)
    player_won = connected_four(board, player)
    opponent_won = connected_four(board, opponent)
    if player_won:
        return GameState.IS_WIN
    elif all(
        check_move_status(board, PlayerAction(col)) == MoveStatus.FULL_COLUMN
        for col in range(board.shape[1])
    ):
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


def get_opponent(player: BoardPiece) -> BoardPiece:
    """
    Returns the opposite player from the current `player`.

    Args:
        player (BoardPiece): The current player.

    Returns:
        BoardPiece: The opponent player.
    """
    if player == PLAYER1:
        return PLAYER2
    elif player == PLAYER2:
        return PLAYER1
    else:
        raise ValueError(f"Invalid player: {player}")


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is accepted as a valid move
    or not, and if not, why.
    Args:
        board (np.ndarray): The current game board.
        column (Any): The column to check (should be PlayerAction).

    Returns:
        MoveStatus: The status of the move (valid, wrong type, out of bounds, or full column).
    """

    # Check Type of column
    if not isinstance(column, PlayerAction):
        return MoveStatus.WRONG_TYPE

    # Check if the column is within bounds
    if column < 0 or column >= BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS

    # Check if the column is full
    if board[0, column] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN

    # If neither of the above conditions are met, the move is valid
    return MoveStatus.IS_VALID
