import numpy as np
from game_utils import (
    BoardPiece,
    BOARD_COLS,
    PlayerAction,
    SavedState,
    apply_player_action,
    MoveStatus,
    check_move_status,
    check_end_state, 
    GameState,
    get_opponent,
    PLAYER1, PLAYER2
)
from .Node import Node


iterationnumber = 20


def mcts_move(
    board: np.ndarray, root_player: BoardPiece, saved_state: SavedState | None,
) -> tuple[PlayerAction, SavedState | None]:
    """
    Perform next move of agent using the Monte Carlo Tree Search (MCTS) algorithm.

    Args:
        board (np.ndarray): The current game board state.
        root_player (BoardPiece): The player making the move.
        saved_state (SavedState | None): The current MCTS tree node state, or None for a new game.

    Returns:
        tuple[PlayerAction, SavedState | None]: The chosen action and the updated tree node (saved state).
    """
    if saved_state is not None and isinstance(saved_state, Node) and np.array_equal(saved_state.state, board) and saved_state.player == root_player:
        print('Use saved state')
        print(saved_state.state)
        root_node = saved_state
       
    else:
        root_node = Node(state=board, player=root_player, parent=None)
    current_node = root_node
    for _ in range(iterationnumber):
        node = current_node
        player = root_player

        # === SELECTION ===
        while node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child()
            player = get_opponent(player)
        # === EXPANSION ===
        if not node.is_terminal and not node.is_fully_expanded():
            action, next_state = expand_to_next_children(player, node)
            next_player = get_opponent(player)
            child_node = node.expand(action, next_state, next_player) 
            node = child_node
            player = next_player
        else:
            break
        
        # === SIMULATION ===
        result = node.result if node.is_terminal else simulate(node, player)

        # === BACKPROPAGATION ===
        backpropagate(node, result)

    # Choose the action of the most visited, for too low iterationnumber it happens that 
    # the children of the current node are empty, so we need to check if there are any children
    # before trying to get the most visited child, as iterationnumber could be too low
    if not current_node.children:
        print("No children found for current_node. Returning a random valid action.")
        valid_moves = [
            col for col in range(BOARD_COLS)
            if check_move_status(current_node.state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        action = np.random.choice(valid_moves)
        return action, current_node
    most_visited = max(current_node.children.items(), key=lambda item: item[1].visits)
    action = most_visited[0]
    apply_player_action(root_node.state, action, root_player)
    saved_state = root_node
    return action, saved_state

def simulate(node: Node, player: BoardPiece) -> dict[BoardPiece, int]:
    """
    Simulate the following game turns from the current node using random play until the game ends.

    Args:
        node (Node): The starting node for simulation.
        player (BoardPiece): The player whose turn it is.
        
    Returns:
        dict[BoardPiece, int]: Mapping of each player to their simulation result score.
    """
    node_state = node.state.copy()

    while check_end_state(node_state, PLAYER1) == GameState.STILL_PLAYING:
        valid_moves = [
            col for col in range(BOARD_COLS)
            if check_move_status(node_state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        if not valid_moves:
            return {PLAYER1: 0, PLAYER2: 0} 
        
        action = np.random.choice(valid_moves)
        apply_player_action(node_state, PlayerAction(action), player)
        player = get_opponent(player)

        result = check_end_state(node_state, PLAYER1)
        if result == GameState.IS_WIN:
            return {PLAYER1: 1, PLAYER2: -1}
        elif result == GameState.IS_LOST:
            return {PLAYER1: -1, PLAYER2: 1}
        elif result == GameState.IS_DRAW:
            return {PLAYER1: 0, PLAYER2: 0}

def backpropagate(node: Node, result: dict[BoardPiece, int]) -> None:
    """
    Backpropagate the result of a simulation, updating win and visit counts.

    Args:
        node (Node): The leaf node where the simulation ended.
        result (dict[BoardPiece, int]): Simulation result mapping for each player.
    """
    current = node
    
    while current is not None:
        current.visits += 1

        for player_name, score in result.items():
            current.wins[player_name] += score
        current = current.parent


def expand_to_next_children(player: BoardPiece, node: Node) -> tuple[PlayerAction, np.ndarray]:
    """
    Select and apply a random untried valid move from the current node's state.

    Args:
        player (BoardPiece): The current player.
        node (Node): The node to expand.

    Returns:
        tuple[PlayerAction, np.ndarray]: The selected action and the resulting game state.
    """
    #node.refresh_children()
    action = np.random.choice(node.untried_actions)
    next_state = node.state.copy()
    apply_player_action(next_state, PlayerAction(action), player)
    return PlayerAction(action), next_state


