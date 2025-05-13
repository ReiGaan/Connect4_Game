import numpy as np
from game_utils import (
    BoardPiece,
    PlayerAction,
    SavedState,
    apply_player_action,
    MoveStatus,
    check_move_status,
    get_opponent,
    check_end_state,
    GameState,
    PLAYER1, PLAYER2
)
from .Node import Node
from game_utils import PLAYER1, PLAYER2

iterationnumber = 2000


def mcts_move(
    board: np.ndarray, root_player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Perform next move of agent using the Monte Carlo Tree Search (MCTS) algorithm.

    Args:
        board (np.ndarray): The current game board state.
        player (BoardPiece): The player making the move.
        saved_state (SavedState | None): The current MCTS tree node state, or None for a new game.

    Returns:
        tuple[PlayerAction, SavedState | None]: The chosen action and the updated tree node (saved state).
    """
    if saved_state is not None and isinstance(saved_state, Node):
        current_node = saved_state
        current_node.parent = None
        current_node.state = board.copy()
        current_node.refresh_children()
    else:
        current_node = Node(state=board.copy(), parent=None)

    for _ in range(iterationnumber):
        node = current_node
        player = root_player

        # === SELECTION ===
        while node.is_fully_expanded() and node.children:
            node = node.best_child(player)
            player = get_opponent(player)

        # === EXPANSION ===
        if not node.is_fully_expanded():
            action, next_state = simulate_next_step(player, node)
            child_node = node.expand(action, next_state) 
            player = get_opponent(player)
            node = child_node
            node.parent = current_node
        else:
            break
        
        # === SIMULATION ===
        result = simulate(node, player)  # Simulate from the new node

        # === BACKPROPAGATION ===
        backpropagate(node, result)

    # Choose the action of the most visited child
    most_visited = max(current_node.children.items(), key=lambda item: item[1].visits)
    action = most_visited[0]
    saved_state = current_node.children[action]  # Save subtree
    return action, saved_state


def simulate(node: Node, player: BoardPiece) -> int:
    """
    Simulate the following game turns from the current node using random play until the game ends.

    Args:
        node (Node): The starting node for simulation.
        root_player (BoardPiece): The player whose turn it is.
        
    Returns:
        int: 1 if player_agent wins, -1 if player_agent loses, 0.5 for a draw.
    """
    node_state = node.state.copy()

    while check_end_state(node_state, PLAYER1) == GameState.STILL_PLAYING:
        valid_moves = [
            col for col in range(node_state.shape[1])
            if check_move_status(node_state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        if not valid_moves:
            break 
        action = np.random.choice(valid_moves)
        playerAction = PlayerAction(action)
        apply_player_action(node_state, playerAction, player)
        player = get_opponent(player)

    result = check_end_state(node_state, PLAYER1)
    if result == GameState.IS_WIN:
        return {PLAYER1: 1, PLAYER2: -1}
    elif result == GameState.IS_LOST:
        return {PLAYER1: -1, PLAYER2: 1}
    elif result == GameState.IS_DRAW:
        return {PLAYER1: 0, PLAYER2: 0}

def backpropagate(node: Node, result: int):
    """
    Backpropagate the result of a simulation, updating win and visit counts.

    Args:
        node (Node): The leaf node where the simulation ended.
        result (int): Simulation result (1 for win, -1 for loss, 0.5 for draw).
        root_player (BoardPiece): The original player making the MCTS move.
    """
    current = node
    
    while current is not None:
        current.visits += 1

        for player_name, score in result.items():
            if score == 1:
                current.wins[player_name] += 1
            elif score == -1:
                current.wins[player_name] -= 1
            elif score == 0:
                current.wins[player_name] += 0

        current = current.parent


def simulate_next_step(player, node):
    """
    Select and apply a random untried valid move from the current node's state.

    Args:
        player (BoardPiece): The current player.
        node (Node): The node to expand.

    Returns:
        tuple[PlayerAction, np.ndarray]: The selected action and the resulting game state.
    """
    action = np.random.choice(node.untried_actions)
    player_action = PlayerAction(action)
    next_state = node.state.copy()
    player_action = PlayerAction(action)
    next_state = node.state.copy()
    apply_player_action(next_state, player_action, player)
    #else:
    #    apply_player_action(next_state, player_action, player)
    if next_state is None:
        print("apply_player_action returned None!")
    return player_action, next_state
