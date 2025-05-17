import numpy as np
import pytest

##TODO: Change board initialization to have real board states

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
            print("Saved node state:\n", saved_state if saved_state else "None")
            print(
                "Valid moves from board:",
                [
                    c
                    for c in range(board.shape[1])
                    if check_move_status(board, c) == MoveStatus.IS_VALID
                ],
            )
            raise

def test_mcts_finds_winning_move_horizontal():
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

def test_mcts_finds_diagonal_win():
    """
    Tests that MCTS finds a diagonal win.
    """
    from game_utils import initialize_game_state, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board = np.array([
        [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [NO_PLAYER, PLAYER1, PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER],
        [PLAYER1, PLAYER2, PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER],
    ])
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(3)

def test_mcts_finds_diagonal_backslash_win():
    """
    Tests that MCTS finds a diagonal win in the "\" direction.
    """
    from game_utils import initialize_game_state, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = np.array([
        [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        [PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER],
        [PLAYER2, PLAYER2, PLAYER2, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER],
    ])

    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(3)


def test_mcts_finds_winning_move_other_player_starts():
    """
    Tests that MCTS selects the winning move when PLAYER2 is the agent and has a win available.
    """
    from game_utils import initialize_game_state, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 2:5] = PLAYER2
    player = PLAYER2
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(1) or action == PlayerAction(5)

def test_mcts_multiple_winning_moves():
    """
    Tests that MCTS picks any of several available winning moves.
    """
    from game_utils import initialize_game_state, PLAYER1, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 1:3] = PLAYER1
    board[5, 4:6] = PLAYER1
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action in [PlayerAction(0), PlayerAction(3), PlayerAction(6)]

def test_mcts_defends_move_vertical():
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

def test_mcts_defends_diagonal():
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 0] = PLAYER2
    board[4, 1] = PLAYER2
    board[3, 2] = PLAYER2
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(3)

def test_mcts_defends_horizontal():
    """
    Tests that MCTS blocks a horizontal win by the opponent.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 2:5] = PLAYER2 
    board[5, 1] = PLAYER1
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(5) 

def test_mcts_defends_move_other_player_starts():
    """
    Tests that MCTS (as PLAYER2) defends the winning move of PLAYER1 when there is one available.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[3:6, 2] = PLAYER1
    board[5, 3:5] = PLAYER2

    player = PLAYER2
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(2)

