import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, check_move_status, MoveStatus
from agent_MCTS.node import node

def mcts_move(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    for i in range(iterationnumber):
        node = board
        # Selection
        while node is fully_expanded and not terminal:
            node = select_child_with_highest_UCT(node)

        # Expansion
        if node is not terminal:
            new_child = expand_random_unvisited_child(node)
            simulation_result = simulate_random_game(new_child)
            
            # Backpropagation
            backpropagate(new_child, simulation_result)
        else:
            # If terminal, simulate from here (though sometimes this step is skipped)
            result = game_result(node)
            backpropagate(node, result)

    return action, saved_state