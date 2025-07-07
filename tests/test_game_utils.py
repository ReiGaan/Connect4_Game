import numpy as np
import pytest


def test_initialize_game_state_shape():
    """
    Checks that the initialized board has shape (6, 7).
    """
    from game_utils import initialize_game_state

    ret = initialize_game_state()
    assert ret.shape == (6, 7)


def test_initialize_game_state_type():
    """
    Checks that the initial board is of the correct type and filled with NO_PLAYER (empty).
    """
    from game_utils import initialize_game_state, BoardPiece, NO_PLAYER

    ret = initialize_game_state()
    assert ret.dtype == BoardPiece and np.all(ret == NO_PLAYER)


def test_pretty_print_empty_board():
    """
    Tests the visual string output for an empty board.
    """
    from game_utils import pretty_print_board, initialize_game_state

    board = initialize_game_state()
    board_str = pretty_print_board(board)

    expected_output = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )

    assert board_str == expected_output


def test_pretty_print_given_board():
    """
    Tests visual string output with a specific, non-empty board state.
    """
    from game_utils import pretty_print_board, NO_PLAYER, PLAYER1, PLAYER2

    given_board = np.array(
        [
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [PLAYER2, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
        ]
    )
    board_str = pretty_print_board(given_board)

    expected_output = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|O O           |\n"
        "|O X X         |\n"
        "|O X O X       |\n"
        "|O O X X O X O |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )

    print("Expected Output:")
    print(expected_output)
    print("Actual Output:")
    print(board_str)

    assert board_str == expected_output


def test_string_to_board_empty_board():
    """
    Checks to convert visual string output back into NumPy array board for empty board.
    """
    from game_utils import string_to_board, NO_PLAYER, BOARD_SHAPE, BoardPiece

    pp_board = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    board_str = string_to_board(pp_board)

    expected_output = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)

    print("Expected Output:")
    print(expected_output)
    print("Actual Output:")
    print(board_str)

    assert np.array_equal(board_str, expected_output)


def test_string_to_board_given_board():
    """
    Checks to convert visual string output back into NumPy array board for not empty board.
    """
    from game_utils import string_to_board, NO_PLAYER, PLAYER1, PLAYER2

    given_board = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|O O           |\n"
        "|O X X         |\n"
        "|O X O X       |\n"
        "|O O X X O X O |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )

    expected_output = np.array(
        [
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [PLAYER2, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
        ]
    )
    board_str = string_to_board(given_board)

    print("Expected Output:")
    print(expected_output)
    print("Actual Output:")
    print(board_str)

    assert np.array_equal(board_str, expected_output)


def test_pretty_print_and_parse_back():
    """
    Checks to initialize game, convert to visual string output and then back into NumPy array board.
    """
    from game_utils import string_to_board, pretty_print_board, initialize_game_state

    board = initialize_game_state()
    board_str = pretty_print_board(board)
    back_board = string_to_board(board_str)

    assert np.array_equal(board, back_board)


def test_player_action_first_action():
    """
    Checks to make an Player action in an empty board.
    """
    from game_utils import (
        initialize_game_state,
        apply_player_action,
        BOARD_SHAPE,
        NO_PLAYER,
        BoardPiece,
        PLAYER1,
        PlayerAction,
    )

    board = initialize_game_state()
    apply_player_action(board, PlayerAction(2), PLAYER1)
    expected_board = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
    expected_board[5, 2] = PLAYER1
    print(expected_board)
    print(board)

    assert np.array_equal(board, expected_board)


def test_player_action_some_full_spots():
    """
    Checks to make an Player action in an not empty board,
    so checks that piece drops into the correct open row above existing pieces.
    """
    from game_utils import (
        apply_player_action,
        NO_PLAYER,
        PLAYER1,
        PLAYER2,
        PlayerAction,
    )

    board = np.array(
        [
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [PLAYER1, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER1, PLAYER1, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
        ]
    )
    expected_board = board.copy()
    expected_board[2, 2] = PLAYER1
    apply_player_action(board, PlayerAction(2), PLAYER1)

    assert np.array_equal(board, expected_board)


def test_player_action_left_edge():
    """
    Checks to make an Player action in an empty board for edge position.
    """
    from game_utils import (
        apply_player_action,
        initialize_game_state,
        PLAYER1,
        PlayerAction,
    )

    board = initialize_game_state()
    apply_player_action(board, PlayerAction(0), PLAYER1)
    assert board[5, 0] == PLAYER1


def test_multiple_actions_in_one_column():
    """
    Checks to make multiple Player action in an empty board in one column.
    """
    from game_utils import (
        apply_player_action,
        initialize_game_state,
        PLAYER1,
        PLAYER2,
        PlayerAction,
    )

    board = initialize_game_state()
    apply_player_action(board, PlayerAction(3), PLAYER1)
    apply_player_action(board, PlayerAction(3), PLAYER2)
    assert board[5, 3] == PLAYER1
    assert board[4, 3] == PLAYER2


def test_connected_four_horizontal():
    """
    Test that a horizontal sequence of four connected PLAYER1 pieces is correctly detected.
    """
    from game_utils import (
        connected_four,
        test_connect_horizontal,
        initialize_game_state,
        PLAYER1,
    )

    board = initialize_game_state()
    board[5, 2:6] = PLAYER1
    assert test_connect_horizontal(board, PLAYER1)
    assert connected_four(board, PLAYER1)


def test_connected_four_vertical():
    """
    Test that a vertical sequence of four connected PLAYER1 pieces is correctly detected.
    """
    from game_utils import (
        connected_four,
        test_connect_vertical,
        initialize_game_state,
        PLAYER1,
    )

    board = initialize_game_state()
    board[2:6, 5] = PLAYER1
    print(board)
    assert test_connect_vertical(board, PLAYER1) == True
    assert connected_four(board, PLAYER1) == True


def test_connected_four_diagonal():
    """
    Checks connect four diagonal ('/') and with Player2.
    """
    from game_utils import (
        test_connect_diagonal,
        connected_four,
        initialize_game_state,
        PLAYER2,
    )

    board = initialize_game_state()

    # diagonle from bottom left to top right
    board[5, 1] = PLAYER2
    board[4, 2] = PLAYER2
    board[3, 3] = PLAYER2
    board[2, 4] = PLAYER2
    assert test_connect_diagonal(board, PLAYER2) and connected_four(board, PLAYER2)


def test_connected_four_diagonal_backslash():
    """
    Checks connect four diagonal ('\') and with Player2.
    """
    from game_utils import (
        test_connect_diagonal,
        connected_four,
        initialize_game_state,
        PLAYER2,
    )

    board2 = initialize_game_state()

    # diagonle from top left to bottom right
    board2[2, 1] = PLAYER2
    board2[3, 2] = PLAYER2
    board2[4, 3] = PLAYER2
    board2[5, 4] = PLAYER2
    assert test_connect_diagonal(board2, PLAYER2) and connected_four(board2, PLAYER2)


def test_connected_four_is_wrong():
    """
    Checks that connect four is false if only 4 pieces on the board but not in  one of the allowed lines.
    """
    from game_utils import (
        connected_four,
        test_connect_vertical,
        test_connect_horizontal,
        test_connect_diagonal,
        initialize_game_state,
        PLAYER1,
    )

    board = initialize_game_state()
    board[2:5, 5] = PLAYER1
    board[2, 4] = PLAYER1
    print(board)
    assert (
        not test_connect_vertical(board, PLAYER1)
        and not test_connect_horizontal(board, PLAYER1)
        and not test_connect_diagonal(board, PLAYER1)
        and not connected_four(board, PLAYER1)
    )


def test_not_possible_player_action_():
    """
    Checks that trying to place a piece in a full column raises a ValueError.
    """
    from game_utils import (
        apply_player_action,
        BOARD_SHAPE,
        NO_PLAYER,
        BoardPiece,
        PLAYER1,
        PLAYER2,
        PlayerAction,
    )

    with pytest.raises(ValueError) as excinfo:
        board = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
        board[:, 2] = PLAYER2
        apply_player_action(board, PlayerAction(2), PLAYER1)

    assert "Column 3 is full." in str(excinfo.value)


def test_check_move_status_Type():
    """
    Checks Move_status for wrong data type.
    """
    from game_utils import initialize_game_state, check_move_status, MoveStatus

    board = initialize_game_state()
    assert check_move_status(board, 0.3) == MoveStatus.WRONG_TYPE
    assert check_move_status(board, "Zwei") == MoveStatus.WRONG_TYPE


def test_check_move_status_Valid():
    """
    Checks Move_status for valid input.
    """
    from game_utils import (
        initialize_game_state,
        check_move_status,
        MoveStatus,
        PlayerAction,
    )

    board = initialize_game_state()
    assert check_move_status(board, PlayerAction(2)) == MoveStatus.IS_VALID


def test_check_move_status_Bounds():
    """
    Checks Move_status for column out of bounds.
    """
    from game_utils import (
        initialize_game_state,
        check_move_status,
        MoveStatus,
        PlayerAction,
    )

    board = initialize_game_state()
    assert check_move_status(board, PlayerAction(7)) == MoveStatus.OUT_OF_BOUNDS


def test_check_move_status_Full():
    """
    Checks Move_status for full column.
    """
    from game_utils import (
        initialize_game_state,
        check_move_status,
        MoveStatus,
        PlayerAction,
        PLAYER1,
    )

    board = initialize_game_state()
    board[:, 2] = PLAYER1
    assert check_move_status(board, PlayerAction(2)) == MoveStatus.FULL_COLUMN


def test_check_end_state_win():
    """
    Checks end_state of game - Win.
    """
    from game_utils import (
        BOARD_SHAPE,
        BoardPiece,
        check_end_state,
        GameState,
        PLAYER1,
        PLAYER2,
        NO_PLAYER,
    )

    board_win = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
    board_win[5, 1:5] = PLAYER1
    assert (check_end_state(board_win, PLAYER1)) == GameState.IS_WIN


def test_check_end_state_loose():
    """
    Checks end_state of game - Win.
    """
    from game_utils import (
        BOARD_SHAPE,
        BoardPiece,
        check_end_state,
        GameState,
        PLAYER1,
        PLAYER2,
        NO_PLAYER,
    )

    board_loose = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
    board_loose[5, 1:5] = PLAYER2
    assert (
        check_end_state(board_loose, PLAYER1)
    ) == GameState.STILL_PLAYING  # as no separate Lost


def test_check_end_state_draw():
    """
    Checks end_state of game - Draw.
    """
    from game_utils import (
        BOARD_SHAPE,
        BoardPiece,
        check_end_state,
        GameState,
        PLAYER1,
        PLAYER2,
        NO_PLAYER,
    )

    board_full = np.array(
        [
            [PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2],
        ]
    )
    assert (check_end_state(board_full, PLAYER2)) == GameState.IS_DRAW and (
        check_end_state(board_full, PLAYER1)
    ) == GameState.IS_DRAW


def test_check_end_state_playing():
    """
    Checks end_state of game - Playing.
    """
    from game_utils import (
        BOARD_SHAPE,
        BoardPiece,
        check_end_state,
        GameState,
        PLAYER1,
        PLAYER2,
        NO_PLAYER,
    )

    board_full = np.array(
        [
            [PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2],
        ]
    )
    board_full[0, :] = NO_PLAYER
    assert (check_end_state(board_full, PLAYER2)) == GameState.STILL_PLAYING


def test_action_type_consistency():
    """
    Test that a PlayerAction instance is considered equal to a numpy int8.
    """
    from game_utils import PlayerAction

    a = PlayerAction(3)
    b = np.int8(3)
    assert a == b
