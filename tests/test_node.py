import numpy as np


def test_get_valid_moves_full_column():
    """
    Checks that if column is full its not in the list of valid
    moves returned by get_valid_moves().
    """
    from game_utils import PLAYER1, initialize_game_state
    from agents.agent_MCTS.node import Node

    board = initialize_game_state()
    board[:, 0] = PLAYER1
    node = Node(board, PLAYER1)
    valid_moves = node.get_valid_moves()
    assert 0 not in valid_moves, "Full column should not be a valid move"


def test_get_valid_moves_edge_cases():
    """
    Test edge cases for the get_valid_moves method.

    This test sets up a board where the first five rows and first four columns are filled by PLAYER1,
    then creates a Node with this board and asserts that only three valid moves remain.

    """
    from game_utils import PLAYER1, PlayerAction
    from agents.agent_MCTS.node import Node

    board = np.zeros((6, 7), dtype=int)
    board[0:5, :4] = PLAYER1
    print(board)
    node = Node(board, PLAYER1)
    expected_moves = [PlayerAction(4), PlayerAction(5), PlayerAction(6)]
    assert set(node.get_valid_moves()) == set(expected_moves)


def test_check_terminal_state_empty_board():
    """
    Test that the Node correctly identifies as not terminal state.
    """
    from game_utils import PLAYER1
    from agents.agent_MCTS.node import Node

    state = np.zeros((6, 7), dtype=int)
    node = Node(state, PLAYER1)
    assert not node.is_terminal


def test_check_terminal_state_winning_state():
    """
    Test that the Node correctly identifies a terminal state (winning).
    """
    from game_utils import PLAYER1, PLAYER2
    from agents.agent_MCTS.node import Node

    state = np.zeros((6, 7), dtype=int)
    node = Node(state, PLAYER1)
    state[0, :4] = PLAYER1
    node = Node(state, PLAYER1)
    assert node.is_terminal
    assert node.result == {PLAYER1: 1, PLAYER2: -1}


def test_expand():
    """Test expanding a node."""
    from game_utils import PLAYER1, PLAYER2, PlayerAction, initialize_game_state
    from agents.agent_MCTS.node import Node

    node = Node(state=initialize_game_state(), player=PLAYER1)
    action = PlayerAction(0)
    next_state = initialize_game_state()
    next_state[5, 0] = PLAYER1
    child = node.expand(action, next_state, PLAYER2)
    assert node.children[action] == child and child.player == PLAYER2


def test_is_fully_expanded():
    """
    Test that a Node becomes fully expanded after all possible actions have been tried.
    """

    from game_utils import PLAYER1, PLAYER2, initialize_game_state, PlayerAction
    from agents.agent_MCTS.node import Node

    state = initialize_game_state()
    node = Node(state=state, player=PLAYER1)

    # Manually create children with dummy next states
    for action in list(node.untried_actions):
        next_state = np.copy(node.state)
        # For test, just mark a cell in first row as next state differentiation
        next_state[0, action] = PLAYER1
        node.expand(PlayerAction(action), next_state, PLAYER2)

    assert (
        node.is_fully_expanded()
    ), f"Node should be fully expanded, but untried_actions remain: {node.untried_actions}"


def test_uct():
    """Test UCT calculation."""
    from game_utils import PLAYER1, PLAYER2, initialize_game_state
    from agents.agent_MCTS.node import Node

    node = Node(state=initialize_game_state(), player=PLAYER1)
    child = Node(initialize_game_state(), PLAYER2, parent=node)
    child.visits = 10
    child.wins[PLAYER1] = 5
    node.visits = 20
    uct_value = node.uct(child)
    assert uct_value > 0


def test_best_child_selects_highest_uct():
    """
    Test that the `best_child` method selects the child node with the highest UCT.
    Test with two children, each having the same number of visits but different win counts for PLAYER1.
    child 1 UCT = 8/10 + sqrt(2 * log(20) / 10) = 1.247
    child 2 UCT = 2/10 + sqrt(2 * log(20) / 10) = 0.747
    """
    from agents.agent_MCTS.node import Node
    from game_utils import initialize_game_state, PlayerAction, PLAYER1, PLAYER2

    root_node = Node(state=initialize_game_state(), player=PLAYER1)
    state1 = root_node.state.copy()
    state2 = root_node.state.copy()
    action1 = PlayerAction(0)
    action2 = PlayerAction(1)
    child1 = Node(state1, PLAYER2, parent=root_node)
    child2 = Node(state2, PLAYER2, parent=root_node)
    child1.visits = 10
    child1.wins[PLAYER1] = 8
    child2.visits = 10
    child2.wins[PLAYER1] = 2
    root_node.children[action1] = child1
    root_node.children[action2] = child2
    root_node.visits = 20
    best = root_node.best_child()
    assert best is child1


def test_best_child_with_unvisited_child():
    """
    Test that `best_child` returns unvisited child node when such a child exists as this ensures
    that prioritizing exploration of unvisited nodes in the tree search algorithm.
    """
    from agents.agent_MCTS.node import Node
    from game_utils import initialize_game_state, PlayerAction, PLAYER1, PLAYER2

    root_node = Node(state=initialize_game_state(), player=PLAYER1)
    state = root_node.state.copy()
    action = PlayerAction(0)
    child = Node(state, PLAYER2, parent=root_node)
    root_node.children[action] = child
    root_node.visits = 1

    assert root_node.best_child() is child
