from .mcts import MCTSAgent
from game_utils import (
    PlayerAction, BoardPiece, MoveStatus, apply_player_action, check_move_status, get_opponent
)
from .node import Node
import numpy as np

class ImprovedMCTSAgent(MCTSAgent):
    def expand_to_next_children(self,
        player: BoardPiece, node: Node
    ) -> tuple[PlayerAction, np.ndarray]:
        """
        Select and apply a valid move from the current node's state.
        This function first checks for immediate win or block opponent's win, and if none are found,
        randomly selects an untried action, this is a more strategic approach than the original random selection.

        Args:
            player (BoardPiece): The current player.
            node (Node): The node to expand.

        Returns:
            tuple[PlayerAction, np.ndarray]: The selected action and the resulting game state.
        """
        opponent = get_opponent(player)
        candidate_moves = []

        for action in node.untried_actions:
            next_state = node.state.copy()
            apply_player_action(next_state, action, player)
            candidate_moves.append((action, next_state))

        # 1. Check for immediate win
        for action, next_state in candidate_moves:
            if Node(next_state, player).check_terminal_state()[0]:
                return action, next_state

        # 2. Check for opponent's immediate win (block it)
        for action, next_state in candidate_moves:
            for opp_action in range(next_state.shape[1]):
                if check_move_status(next_state, PlayerAction(opp_action)) == MoveStatus.IS_VALID:
                    opp_state = next_state.copy()
                    apply_player_action(opp_state, PlayerAction(opp_action), opponent)
                    if Node(opp_state, opponent).check_terminal_state()[0]:
                        return action, next_state

        #Otherwise still, pick randomly as before
        action = np.random.choice(list(node.untried_actions))
        next_state = node.state.copy()
        apply_player_action(next_state, action, player)
        return action, next_state

    def __call__(self, board, player, saved_state, *args):
        return self.mcts_move(board, player, saved_state, *args)