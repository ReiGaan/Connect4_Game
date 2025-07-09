import numpy as np
from agents.agent_MCTS.hierarchical_mcts import HierarchicalMCTSAgent
from agents.agent_MCTS.node import Node

from game_utils import (
    PLAYER1,
    PLAYER2,
    PlayerAction,
    BoardPiece,
    MoveStatus,
    apply_player_action,
    check_move_status,
)


def empty_board():
    # Standard Connect4 board: 6 rows x 7 columns
    return np.zeros((6, 7), dtype=np.int8)


def agent():
    return HierarchicalMCTSAgent(
        iterationnumber=10, max_depth_for_minmax=4, max_simulation_depth=10
    )


def test_count_n_in_a_row_horizontal():
    """
    Should count a horizontal n-in-a-row for PLAYER1
    """
    ag = agent()
    board = empty_board()
    board[0, 0:3] = PLAYER1
    assert ag.count_n_in_a_row(board, PLAYER1, 3) == 1


def test_count_n_in_a_row_vertical():
    """
    Should count a vertical n-in-a-row for PLAYER2
    """
    ag = agent()
    board = empty_board()
    board[0:3, 0] = PLAYER2
    assert ag.count_n_in_a_row(board, PLAYER2, 3) == 1


def test_count_n_in_a_row_diagonal():
    """
    Should count a diagonal n-in-a-row for PLAYER1
    """
    ag = agent()
    board = empty_board()
    for i in range(3):
        board[i, i] = PLAYER1
    assert ag.count_n_in_a_row(board, PLAYER1, 3) == 1


def test_expand_to_next_children_immediate_win():
    """
    Agent should take immediate winning move if available
    """
    ag = agent()
    board = empty_board()
    board[5, 0:3] = PLAYER1
    node = Node(board, PLAYER1)
    node.untried_actions = [
        PlayerAction(i)
        for i in range(7)
        if check_move_status(board, PlayerAction(i)) == MoveStatus.IS_VALID
    ]
    action, next_state = ag.expand_to_next_children(PLAYER1, node)
    assert (next_state[5, 3] == PLAYER1) or (next_state[4, 3] == PLAYER1)


def test_expand_to_next_children_block_opponent():
    """
    Agent should block opponent's immediate win
    """
    ag = agent()
    board = empty_board()
    board[5, 0:3] = PLAYER2
    node = Node(board, PLAYER1)
    node.untried_actions = [
        PlayerAction(i)
        for i in range(7)
        if check_move_status(board, PlayerAction(i)) == MoveStatus.IS_VALID
    ]
    action, next_state = ag.expand_to_next_children(PLAYER1, node)
    assert int(action) == 3
