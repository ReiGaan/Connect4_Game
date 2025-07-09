import numpy as np
from game_utils import (
    BoardPiece,
    PlayerAction,
    SavedState,
    apply_player_action,
    MoveStatus,
    check_move_status,
    get_opponent,
    PLAYER1,
    PLAYER2,
    check_end_state,
)
from metrics.metrics import GameMetrics
from .node import Node


class MCTSAgent:
    """
    An agent that uses the Monte Carlo Tree Search (MCTS) algorithm to select actions in a turn-based game.

    The MCTSAgent explores possible future game states by building a search tree through repeated simulations,
    allowing it to make statistically-informed decisions without explicit strategy coding.

    The agent follows the four standard MCTS phases:
        1. Selection: Traverse the tree by selecting the best child nodes using a policy (e.g., UCB1).
        2. Expansion: Add a new child node corresponding to an untried action.
        3. Simulation: Play out the game from the new node using random moves.
        4. Backpropagation:  Propagate the result of the simulation back up the tree to update statistics.

    Attributes:
        iteration_count (int): Number of MCTS iterations to perform per move.

    Methods:
        mcts_move(board, root_player, saved_state, player_name, metrics=None):
            Executes MCTS from the current board state and returns the chosen action and updated state.
        selection_process(node):
            Traverses the tree to find the next node to expand.
        expansion(node, player):
            Expands the node by adding a new child for an untried move.
        simulate(node, player):
            Simulates a random play-out from a given node until a terminal state is reached.
        backpropagate(node, result):
            Updates the tree statistics based on simulation results.
        expand_to_next_children(player, node):
            Chooses the next untried action for expansion, prioritizing winning or blocking moves.
        __call__(board, player, saved_state, *args):
            Delegates to mcts_move to make the agent callable like a function.
    """

    def __init__(self, iterationnumber: int = 100):
        """
        Initialize the MCTS agent with a specified number of iterations.

        Args:
            iterationnumber (int): The number of iterations for MCTS.
        """
        self.iterationnumber = iterationnumber

    def mcts_move(
        self,
        board: np.ndarray,
        root_player: BoardPiece,
        saved_state: SavedState | None,
        player_name: str | None = None,
        metrics: GameMetrics | None = None,
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
        if (
            saved_state is not None
            and isinstance(saved_state, Node)
            and np.array_equal(saved_state.state, board)
            and saved_state.player == root_player
        ):
            root_node = saved_state

        else:
            root_node = Node(state=board, player=root_player, parent=None)
        current_node = root_node
        for _ in range(self.iterationnumber):
            node = current_node
            player = root_player

            # === SELECTION ===
            node = self.selection_process(node)
            # === EXPANSION ===
            node = self.expansion(node, player)

            # === SIMULATION ===
            result = node.result if node.is_terminal else self.simulate(node, player)

            # === BACKPROPAGATION ===
            self.backpropagate(node, result)

        # fallback to random valid action if no children exist.
        if not current_node.children:
            print(
                "No children found for current_node. Returning a random valid action."
            )
            valid_moves = current_node.get_valid_moves()
            action = np.random.choice(valid_moves)
            return action, current_node
        most_visited = max(
            current_node.children.items(), key=lambda item: item[1].visits
        )
        action = most_visited[0]
        apply_player_action(root_node.state, action, root_player)
        saved_state = root_node
        return action, saved_state

    def selection_process(self, node):
        """
        Traverses the tree from the given node by repeatedly selecting the best child node
        until a node is found that is either not fully expanded or is a terminal node.
        Args:
            node: The starting node for the selection process.
        Returns:
            The first node encountered that is either not fully expanded or is a terminal node.
        """

        while node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child()
        return node

    def expansion(self, node, player):
        """
        Expands the given node if it is not terminal and not fully expanded.
        It selects the next possible action and state, creates a new child node for that action,
        and returns the newly created child node. If the node is terminal or fully expanded, it returns the original node.

        Args:
            node: The current node in the MCTS tree to be expanded.
            player: The player for whom the expansion is being performed.

        Returns:
            The expanded child node if expansion occurred, otherwise the original node.
        """
        if not node.is_terminal and not node.is_fully_expanded():
            action, next_state = self.expand_to_next_children(player, node)
            next_player = get_opponent(player)
            child_node = node.expand(action, next_state, next_player)
            node = child_node
        return node

    def simulate(self, node: Node, player: BoardPiece) -> dict[BoardPiece, int]:
        """
        Simulate the following game turns from the current node using random play until the game ends.

        Args:
            node (Node): The starting node for simulation.
            player (BoardPiece): The player whose turn it is.

        Returns:
            dict[BoardPiece, int]: Mapping of each player to their simulation result score.
        """
        node_state = node.state.copy()
        # Note: This simulation assumes games will always terminate.
        while True:
            valid_moves = [
                col
                for col in range(node_state.shape[1])
                if check_move_status(node_state, PlayerAction(col))
                == MoveStatus.IS_VALID
            ]
            if not valid_moves:
                return {PLAYER1: 0, PLAYER2: 0}

            action = np.random.choice(valid_moves)
            apply_player_action(node_state, PlayerAction(action), player)

            state = check_end_state(node_state, player)
            if state.name == "IS_WIN":
                return {player: 1, get_opponent(player): -1}
            elif state.name == "IS_DRAW":
                return {PLAYER1: 0, PLAYER2: 0}

            player = get_opponent(player)

    def backpropagate(self, node: Node, result: dict[BoardPiece, int]) -> None:
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

    def expand_to_next_children(
        self, player: BoardPiece, node: Node
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

        for action in node.untried_actions:
            next_state = node.state.copy()
            apply_player_action(next_state, action, player)

        # Otherwise, pick randomly as before
        action = np.random.choice(list(node.untried_actions))
        next_state = node.state.copy()
        apply_player_action(next_state, action, player)
        return action, next_state

    def __call__(self, board, player, saved_state, *args):
        return self.mcts_move(board, player, saved_state, *args)
