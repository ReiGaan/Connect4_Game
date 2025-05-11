import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, apply_player_action, MoveStatus, check_move_status, get_opponent, check_end_state, GameState
from .Node import Node

iterationnumber = 1000

def mcts_move(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    if saved_state is not None and isinstance(saved_state, Node):
        current_node = saved_state
        current_node.parent = None 
        current_node.state = board.copy() 
        current_node.untried_actions = current_node.get_valid_moves()  
    else:
        current_node = Node(state=board.copy(), parent=None)

    for _ in range(iterationnumber):
        node = current_node
        current_player = player

        # === SELECTION ===
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # === EXPANSION ===
        if not node.is_fully_expanded():
            action, next_state = simulate_next_step(player, node)
            next_node = node.expand(action, next_state)
            oppponent = get_opponent(player)
            node = next_node
            current_player = oppponent
        else: 
            break

        # === SIMULATION === (on plain board)
        result = simulate(node, current_player, player)

        # === BACKPROPAGATION ===
        backpropagate(node, result, player)

    # Choose the action of the most visited child
    most_visited = max(
        current_node.children.items(), key=lambda item: item[1].visits
    )
    action = most_visited[0]
    saved_state = current_node.children[action]  # Save subtree
    return action, saved_state

def simulate(node: Node, current_player: BoardPiece, player_agent: BoardPiece) -> int:
    node_state = node.state.copy()
    #current_player is opponent
    while True:
        valid_moves = [
            col for col in range(node_state.shape[1])
            if check_move_status(node_state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        if not valid_moves:
            return 0.5 
        action = np.random.choice(valid_moves)
        playerAction = PlayerAction(action)
        apply_player_action(node_state, playerAction, current_player)

        result = check_end_state(node_state, current_player)
        if result == GameState.IS_WIN:
            return 1 if current_player == player_agent else -1
        elif result == GameState.IS_DRAW:
            return 0.5
        elif result == GameState.STILL_PLAYING:
            current_player = get_opponent(current_player)
            continue

def backpropagate(node: Node, result: int, root_player: BoardPiece):
    current = node
    player = get_opponent(root_player)  # simulation started from opponent's turn

    while current is not None:
        current.visits += 1

        if result == 1 and player == root_player:
            current.wins += 1
        elif result == -1 and player != root_player:
            current.wins += 1
        elif result == 0:
            current.wins += 0.5

        player = get_opponent(player)
        current = current.parent

def simulate_next_step(player, node):
    action = np.random.choice(node.untried_actions)
    player_action = PlayerAction(action)
    next_state = node.state.copy()
    if check_move_status(next_state, player_action) != MoveStatus.IS_VALID:
        print(f"Invalid move: {action}")
        print(f"Untried: {node.untried_actions}")
        valide_moves = node.get_valid_moves()
        print(f"Valid moves: {valide_moves}")   
        action = np.random.choice(valide_moves)
        player_action = PlayerAction(action)    
        next_state = node.state.copy()
        apply_player_action(next_state, player_action, player)
    else: 
        apply_player_action(next_state, player_action, player)
    if next_state is None:
        print("apply_player_action returned None!")
    return player_action,next_state


            
                
              