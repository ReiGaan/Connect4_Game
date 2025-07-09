import numpy as np

from agents.agent_MCTS.mcts import MCTSAgent
from agents.agent_MCTS.node import Node
from game_utils import (
    initialize_game_state,
    PLAYER1,
    PLAYER2,
    PlayerAction,
    apply_player_action,
    get_opponent,
)


def test_child_node_player_after_expansion():
    board = initialize_game_state()
    root_node = Node(state=board, player=PLAYER1)

    action = PlayerAction(0)
    next_state = board.copy()
    apply_player_action(next_state, action, PLAYER1)
    child_node = root_node.expand(action, next_state, PLAYER2)

    root_node.untried_actions = set()

    agent = MCTSAgent(iterationnumber=1)
    agent.mcts_move(board.copy(), PLAYER1, root_node)

    assert child_node.children, "Child node should have a child after expansion"
    grandchild = next(iter(child_node.children.values()))
    expected_player = get_opponent(child_node.player)
    assert grandchild.player == expected_player
