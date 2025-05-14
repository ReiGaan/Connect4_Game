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
    assert ret.dtype == BoardPiece
    assert np.all(ret == NO_PLAYER)


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

    print("Expected Output:")
    print(expected_output)
    print("Actual Output:")
    print(board_str)

    assert board_str == expected_output


def test_pretty_printgiven_board():
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

    assert board_str.all() == expected_output.all()


def test_pp_back():
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
    )

    board = initialize_game_state()
    apply_player_action(board, 2, PLAYER1)
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
    from game_utils import apply_player_action, NO_PLAYER, PLAYER1, PLAYER2

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
    apply_player_action(board, 2, PLAYER1)

    assert np.array_equal(board, expected_board)


def test_player_action_left_edge():
    """
    Checks to make an Player action in an empty board for edge position.
    """
    from game_utils import apply_player_action, initialize_game_state, PLAYER1

    board = initialize_game_state()
    apply_player_action(board, 0, PLAYER1)
    assert board[5, 0] == PLAYER1


def test_multiple_actions_in_one_column():
    """
    Checks to make multiple Player action in an empty board in one column.
    """
    from game_utils import apply_player_action, initialize_game_state, PLAYER1, PLAYER2

    board = initialize_game_state()
    apply_player_action(board, 3, PLAYER1)
    apply_player_action(board, 3, PLAYER2)
    assert board[5, 3] == PLAYER1
    assert board[4, 3] == PLAYER2


def test_connected_four_horizontal():
    """ """
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
    """ """
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
    Checks connect four diagonal and with Player2 this time.
    """
    from game_utils import (
        test_connect_diagonal,
        connected_four,
        initialize_game_state,
        PLAYER2,
    )

    board = initialize_game_state()
    board2 = initialize_game_state()
    # diagonle from bottom left to top right
    board[5, 1] = PLAYER2
    board[4, 2] = PLAYER2
    board[3, 3] = PLAYER2
    board[2, 4] = PLAYER2
    # diagonle from top left to bottom right
    board2[2, 1] = PLAYER2
    board2[3, 2] = PLAYER2
    board2[4, 3] = PLAYER2
    board2[5, 4] = PLAYER2
    assert test_connect_diagonal(board, PLAYER2) == True
    assert connected_four(board, PLAYER2) == True
    assert test_connect_diagonal(board2, PLAYER2) == True
    assert connected_four(board2, PLAYER2) == True


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
    assert test_connect_vertical(board, PLAYER1) == False
    assert test_connect_horizontal(board, PLAYER1) == False
    assert test_connect_diagonal(board, PLAYER1) == False
    assert connected_four(board, PLAYER1) == False


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
    )

    with pytest.raises(ValueError) as excinfo:
        board = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
        board[:, 2] = PLAYER2
        apply_player_action(board, 2, PLAYER1)

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
    from game_utils import initialize_game_state, check_move_status, MoveStatus, PlayerAction

    board = initialize_game_state()
    assert check_move_status(board, PlayerAction(2)) == MoveStatus.IS_VALID


def test_check_move_status_Bounds():
    """
    Checks Move_status for column out of bounds.
    """
    from game_utils import initialize_game_state, check_move_status, MoveStatus, PlayerAction

    board = initialize_game_state()
    assert check_move_status(board, PlayerAction(7)) == MoveStatus.OUT_OF_BOUNDS


def test_check_move_status_Full():
    """
    Checks Move_status for full column.
    """
    from game_utils import initialize_game_state, check_move_status, MoveStatus, PlayerAction, PLAYER1

    board = initialize_game_state()
    board[:, 2] = PLAYER1
    assert check_move_status(board, PlayerAction(2)) == MoveStatus.FULL_COLUMN


def test_check_end_state():
    """
    Checks end_state of game.
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
    board_loose = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
    board_loose[5, 1:5] = PLAYER2
    assert (check_end_state(board_loose, PLAYER1)) == GameState.IS_LOST
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
    assert (check_end_state(board_full, PLAYER2)) == GameState.IS_DRAW
    assert (check_end_state(board_full, PLAYER1)) == GameState.IS_DRAW
    board_full[0, :] = NO_PLAYER
    assert (check_end_state(board_full, PLAYER2)) == GameState.STILL_PLAYING


def test_mcts_never_plays_illegal_move():
    from game_utils import (
        BoardPiece,
        MoveStatus,
        check_move_status,
        NO_PLAYER,
        PLAYER1,
        PLAYER2,
    )
    from agents.agent_MCTS.mcts import mcts_move

    num_trials = 8

    for i in range(num_trials):
        board = np.array(
            [
                [NO_PLAYER, PLAYER1, PLAYER2, NO_PLAYER, PLAYER2, PLAYER1, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1],
                [PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1],
                [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2],
            ]
        )

        player = BoardPiece(1)
        saved_state = None

        action, saved_state = mcts_move(board.copy(), player, saved_state, iterationnumber=10)

        try:
            action, saved_state = mcts_move(board.copy(), player, saved_state)
            move_status = check_move_status(board, action)
            assert move_status == MoveStatus.IS_VALID
        except AssertionError:
            print(f"Test failed on iteration {i}")
            print("Board before move:\n", board)
            print("Action selected:", action)
            print("Saved node state:\n", saved_state.state if saved_state else "None")
            print(
                "Valid moves from board:",
                [
                    c
                    for c in range(board.shape[1])
                    if check_move_status(board, c) == MoveStatus.IS_VALID
                ],
            )
            raise

def test_check_terminal_state_empty_board():
    from game_utils import PLAYER1, PLAYER2
    from agents.agent_MCTS.Node import Node
    state = np.zeros((6, 7), dtype=int)
    node = Node(state, PLAYER1)
    assert not node.is_terminal

def test_check_terminal_state_winning_state():
    from game_utils import PLAYER1, PLAYER2
    from agents.agent_MCTS.Node import Node
    state = np.zeros((6, 7), dtype=int)
    node = Node(state, PLAYER1)
    state[0, :4] = PLAYER1
    node = Node(state, PLAYER1)
    assert node.is_terminal
    assert node.result == {PLAYER1: 1, PLAYER2: -1}


def test_mcts_finds_winning_move():
    """
    Tests that MCTS selects the winning move when there is one available.
    """
    from game_utils import initialize_game_state, PLAYER1, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 2:5] = PLAYER1
    print(board)
    player = PLAYER1
    saved_state = None
    
    action, _ = mcts_move(board.copy(), player, saved_state)
    print(action)
    assert action == PlayerAction(1) or action == PlayerAction(5) 

def test_mcts_defends_move():
    """
    Tests that MCTS defends the winning move of the opponent when there is one available
    and there is a way to defend it.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[3:6, 2] = PLAYER2
    board[5, 3:5] = PLAYER1

    player = PLAYER1
    saved_state = None
    
    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(2)

def test_edge_cases():
    from game_utils import PLAYER1
    from agents.agent_MCTS.Node import Node
    board = np.zeros((6, 7), dtype=int)
    board[0:5,:4] = PLAYER1 
    print(board)
    node = Node(board, PLAYER1)
    assert len(node.get_valid_moves()) == 3 
    
    
def test_backpropagation(): 
    from agents.agent_MCTS.Node import Node
    from agents.agent_MCTS.mcts import backpropagate
    from game_utils import PLAYER1, PLAYER2
    
    root_node = Node(state=np.zeros((6, 7)), parent=None, player=PLAYER1)   
    child_node_1 = Node(state=np.zeros((6, 7)), parent=root_node, player=PLAYER2)  
    child_node_2 = Node(state=np.zeros((6, 7)), parent=child_node_1, player=PLAYER1)    
    
    root_node.visits = 0
    child_node_1.visits = 0
    child_node_2.visits = 0
    
    final_result = {PLAYER1: 1, PLAYER2: -1}
    
    backpropagate(child_node_2, final_result)
    
    assert root_node.visits == 1 and root_node.wins[PLAYER1] == 1
    
def test_get_valid_moves_full_column():
    from game_utils import PLAYER1, initialize_game_state
    from agents.agent_MCTS.Node import Node
    board = initialize_game_state()
    board[:, 0] = PLAYER1  
    node = Node(board, PLAYER1)
    valid_moves = node.get_valid_moves()
    assert 0 not in valid_moves, "Full column should not be a valid move"
    
def test_expand():
    """Test expanding a node."""
    from game_utils import PLAYER1, PLAYER2, PlayerAction, initialize_game_state
    from agents.agent_MCTS.Node import Node
    node = Node(state=initialize_game_state(), player=PLAYER1)
    action = PlayerAction(0)
    next_state = initialize_game_state()
    next_state[5, 0] = PLAYER1  
    child = node.expand(action, next_state, PLAYER2)
    assert node.children[action] == child and child.player == PLAYER2

def test_is_fully_expanded():
    """Test if a node is fully expanded."""
    from game_utils import PLAYER1, PLAYER2, initialize_game_state
    from agents.agent_MCTS.Node import Node
    node = Node(state=initialize_game_state(), player=PLAYER1)
    for action in node.untried_actions:
        next_state = initialize_game_state()
        next_state[5, action] = PLAYER1
        node.expand(action, next_state, PLAYER2)
    assert node.is_fully_expanded() 

def test_uct():
    """Test UCT calculation."""
    from game_utils import PLAYER1, PLAYER2, initialize_game_state
    from agents.agent_MCTS.Node import Node
    node = Node(state=initialize_game_state(), player=PLAYER1)
    child = Node(initialize_game_state(), PLAYER2, parent=node)
    child.visits = 10
    child.wins[PLAYER1] = 5
    node.visits = 20
    uct_value = node.uct(child)
    assert uct_value > 0 

def test_best_child():
    """Test selecting the best child node based on UCT."""
    from game_utils import PLAYER1, PLAYER2, PlayerAction, initialize_game_state
    from agents.agent_MCTS.Node import Node
    node = Node(state=initialize_game_state(), player=PLAYER1)
    for i in range(3):
        next_state = initialize_game_state()
        next_state[5, i] = PLAYER1
        child = node.expand(PlayerAction(i), next_state, PLAYER2)
        child.visits = i + 1 
    best_child = node.best_child()
    print(best_child.visits)
    assert best_child.visits == 3  

