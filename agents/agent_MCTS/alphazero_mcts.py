from .mcts import MCTSAgent
from game_utils import (
    PlayerAction, PLAYER1, PLAYER2,  BoardPiece, MoveStatus, apply_player_action, check_move_status, get_opponent
)
from .node import Node
"""
Needed from Network:
function that wraps around your neural network and applies it to a board state:
method policy_value(state)
(policy_dict, value) = policy_value_fn(state)
policy_dict: {action: probability} only legal actions
value: a scalar between 0–1 (e.g., probability of win for current player)
"""

class AlphazeroMCTSAgent(MCTSAgent):
    """MCTS agent for Connect4 useable for alphazero implementation.
    This agent changes the base MCTSAgent by several parts:
    - Use prior knowledge for best child choose and so change UCT to PUCT function.
    - ...
    - ...
    Functions:
        mcts_move(board, root_player, saved_state):
            Performs the MCTS move based on the current board state and player.
        __call__(board, player, saved_state, *args):
            Returns the agent's move using MCTS.
    Attributes:
        iterationnumber (int): Number of MCTS iterations per move.
    """
    def __init__(self, policy_value: callable, iterationnumber=100):
        super().__init__(iterationnumber)
        self.policy_value = policy_value

    def simulate(self, node: Node, player: BoardPiece):
        _, value = self.policy_value(node.state)
        # Return value from the root player’s perspective
        return {PLAYER1: value, PLAYER2: 1 - value}
    
    def selection_process(self, node):
        """
        Traverses the tree from the given node by repeatedly selecting the best child node based on prior knowledge
        until a node is found that is either not fully expanded or is a terminal node.
        Args:
            node: The starting node for the selection process.
        Returns:
            The first node encountered that is either not fully expanded or is a terminal node.
        """
        
        while node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child_based_prior_knowledge()
        return node


    def expansion(self, node, player):
        if node.is_terminal or node.is_fully_expanded():
            return node

        policy, _ = self.policy_value(node.state)
        for action in node.untried_actions.copy(): 
            next_state = node.state.copy()
            apply_player_action(next_state, action, player)
            prior = policy.get(action, 1e-5) 
            next_player = get_opponent(player)
            node.expand(action, next_state, next_player, prior=prior)
        return node
