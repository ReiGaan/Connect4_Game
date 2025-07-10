from .mcts import MCTSAgent
from game_utils import (
    PlayerAction,
    PLAYER1,
    PLAYER2,
    BoardPiece,
    MoveStatus,
    apply_player_action,
    check_move_status,
    get_opponent,
)
from .node import Node


class AlphazeroMCTSAgent(MCTSAgent):
    """MCTS agent for Connect4 useable for alphazero implementation.
    This agent changes the base MCTSAgent:
    - Use prior knowledge for best child choose and so change UCT to PUCT function.
    Functions:
        mcts_move(board, root_player, saved_state):
            Performs the MCTS move based on the current board state and player.
        __call__(board, player, saved_state, *args):
            Returns the agent's move using MCTS.
    Attributes:
        iterationnumber (int): Number of MCTS iterations per move.
    """

    def __init__(self, policy_value: callable, iterationnumber=100):
        """
        Initializes the AlphaZero MCTS agent.

        Args:
            policy_value (callable): A function that takes a board state and the
                current player and returns a tuple of
                (policy: Dict[action, prior], value: float).
            iterationnumber (int): Number of MCTS iterations per move.
        """
        super().__init__(iterationnumber)
        self.policy_value = policy_value

    def simulate(self, node: Node, player: BoardPiece):
        """
        Simulates the value of a node using the policy-value function.

        Args:
            node (Node): The current node in the tree.
            player (BoardPiece): The player performing the simulation.

        Returns:
            Dict[BoardPiece, float]: A dictionary mapping players to value estimates from the root player's perspective.
        """
        if node.is_terminal:
            return node.result

        _, value = self.policy_value(node.state, node.player)
        # The network's value is from the perspective of the player to move.
        # Convert it into per‑player scores for backpropagation.
        return {player: value, get_opponent(player): -value}

    def selection_process(self, node: Node) -> Node:
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

    def expansion(self, node: Node, player: BoardPiece) -> Node:
        """
        Expands the given node by adding child nodes for each untried action.
        For each untried action from the current node, this method:
        If the node is terminal or already fully expanded, it is returned unchanged.
        Args:
            node: The current MCTS node to expand.
            player: The player to apply actions for at this node.
        Returns:
            The expanded node (or the original node if terminal or fully expanded).
        """
        if node.is_terminal or node.is_fully_expanded():
            return node

        policy, _ = self.policy_value(node.state, node.player)
        for action in node.untried_actions.copy():
            next_state = node.state.copy()
            apply_player_action(next_state, action, player)
            prior = policy.get(action, 1e-5)
            next_player = get_opponent(player)
            node.expand(action, next_state, next_player, prior=prior)
        return node
