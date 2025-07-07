"""
This module defines the Node class for use in Monte Carlo Tree Search (MCTS).

Classes:
    Node: Represents a node in the MCTS tree, encapsulating the game state,
          player to move, parent and child nodes, visit and win statistics,
          and methods for expansion, terminal state checking, and UCT/PUCT-based child selection.

Dependencies:
    - numpy
    - game_utils: connected_four, PlayerAction, check_move_status, MoveStatus,
                  BOARD_COLS, BoardPiece, PLAYER1, PLAYER2, NO_PLAYER
"""

import numpy as np
from game_utils import (
    connected_four,
    PlayerAction,
    check_move_status,
    MoveStatus,
    BOARD_COLS,
    BoardPiece,
    PLAYER1,
    PLAYER2,
    NO_PLAYER,
)


class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS).

    Attributes:
        state (np.ndarray): The game state associated with this node.
        player (BoardPiece): The player who is to move at this node.
        parent (Node | None): The parent node in the search tree.
        children (dict): A dictionary mapping actions to child nodes.
        visits (int): Number of times this node has been visited.
        wins (dict): Simulation result scores for this node winning for both player (Player == key).
        untried_actions (list): List of actions not yet tried from this state.
        is_terminal (bool): Whether this node is a terminal state.
        result (dict): The result of the game, if node is terminal.
        prior (float): Optional prior probability for the node (for PUCT).
    """

    def __init__(
        self, state: np.ndarray, player: BoardPiece, parent=None, prior: float = 0.0
    ):
        """
        Initialize a Node for MCTS.

        Args:
            state (np.ndarray): The game state.
            player (BoardPiece): The player to move at this node.
            parent (Node | None): The parent node.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Expected state to be a np.ndarray, got {type(state)}")

        self.state = state.copy()
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = {PLAYER1: 0, PLAYER2: 0}
        self.untried_actions = set(self.get_valid_moves())
        self.is_terminal, self.result = self.check_terminal_state()
        self.prior = prior

    def check_terminal_state(self) -> tuple[bool, dict | None]:
        """
        Check whether the current node represents a terminal state in the game.

        Returns:
            tuple:
                - bool: True if the current state is terminal, False otherwise.
                - dict or None: A dictionary with the result of the game for each player
                if the state is terminal, or None if not terminal.
        """
        if self._is_win(PLAYER1):
            return True, {PLAYER1: 1, PLAYER2: -1}
        if self._is_win(PLAYER2):
            return True, {PLAYER1: -1, PLAYER2: 1}
        if self._is_draw():
            return True, {PLAYER1: 0, PLAYER2: 0}
        return False, None

    def _is_win(self, player: BoardPiece) -> bool:
        """
        Check if the given player has won.

        Args:
            player (BoardPiece): The player to check for a win.

        Returns:
            bool: True if the player has won, False otherwise.
        """
        return connected_four(self.state, player)

    def _is_draw(self) -> bool:
        """
        Check if the game is a draw.
        A draw occurs when there are no valid moves left and no player has won.
        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        return bool(np.all(self.state != NO_PLAYER))

    def get_valid_moves(self) -> list[PlayerAction]:
        """
        Compute the list of valid moves from the current Node.

        Returns:
            list[PlayerAction]: A list of column indices that are valid moves.
        """
        return [
            PlayerAction(col)
            for col in range(BOARD_COLS)
            if check_move_status(self.state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]

    def expand(
        self,
        action: PlayerAction,
        next_state: np.ndarray,
        next_player: BoardPiece,
        prior: float = 0.0,
    ) -> "Node":
        """
        Expand the current node by adding a child node for the given action.

        Args:
            action (PlayerAction): The action that leads to the child state.
            next_state (np.ndarray): The resulting state from applying the action.
            next_player (BoardPiece): The player who will move next.

        Returns:
            child_node (Node): The newly created child node.
        """
        child_node = Node(next_state, next_player, parent=self, prior=prior)
        self.children[action] = child_node
        self.untried_actions.discard(action)
        return child_node

    def is_fully_expanded(self) -> bool:
        """
        Check whether all possible actions from this node have been tried.

        Returns:
            bool: True if no untried actions remain, False otherwise.
        """
        return len(self.untried_actions) == 0

    def uct(self, child: "Node", exploration_param: float = np.sqrt(2)) -> float:
        """
        Calculate the Upper Confidence Bound (UCT) score for a child node.

        Args:
            child (Node): The child node to evaluate.
            exploration_param (float): The exploration parameter for UCT.
        Returns:
            float: The UCT score for the node.
        """
        if child.visits == 0:
            return float("inf")
        exploitation = child.wins[self.player] / child.visits
        exploration = exploration_param * np.sqrt(
            np.log(self.visits + 1) / child.visits
        )
        return exploitation + exploration

    def puct(self, child: "Node", exploration_param: float = np.sqrt(2)) -> float:
        """
        Calculate the PUCT score for a child node.

        Args:
            prior (float): Prior probability P(s,a) from policy network.
            child (Node): The child node to evaluate.
            exploration_param (float): The exploration parameter for PUCT.
        Returns:
            float: The PUCT score for the node.
        """
        if child.visits == 0:
            return float("inf")
        exploitation = child.wins[self.player] / child.visits
        exploration = (
            exploration_param * child.prior * np.sqrt(self.visits) / (1 + child.visits)
        )
        return exploitation + exploration

    def best_child(self) -> "Node":
        """
        Select the child node with the highest UCT score.

        Returns:
            best_child (Node): The child node with the highest UCT value.
        """
        best_uct_value = float("-inf")
        best_node = None

        for action, child in self.children.items():
            score = self.uct(child)

            if score > best_uct_value:
                best_uct_value = score
                best_node = child
        if best_node is None:
            raise ValueError("No best child found: this node has no children.")
        return best_node

    def best_child_based_prior_knowledge(self) -> "Node":
        """
        Select the child node with the highest PUCT score.

        Returns:
            best_child (Node): The child node with the highest PUCT value.
        """
        best_puct_value = float("-inf")
        best_node = None

        for action, child in self.children.items():
            score = self.puct(child)

            if score > best_puct_value:
                best_puct_value = score
                best_node = child
        if best_node is None:
            raise ValueError("No best child found: this node has no children.")
        return best_node
