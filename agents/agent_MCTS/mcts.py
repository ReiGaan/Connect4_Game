"""
This module implements the Monte Carlo Tree Search (MCTS) algorithm for game-playing agents.
Functions:
mcts_move(board, root_player, saved_state, iterationnumber=iterationnumber)
    Perform the next move using the MCTS algorithm, returning the chosen action and updated tree node.
simulate(node, player)
    Simulate a random playout from the given node until the game ends, returning the result.
    Backpropagate the simulation result up the tree, updating win and visit counts.
expand_to_next_children(player, node)
    Expand the current node by selecting and applying a valid move, prioritizing immediate wins or blocks.
Constants:
iterationnumber: Default number of MCTS iterations per move.
Imports:
numpy, game_utils (BoardPiece, PlayerAction, SavedState, apply_player_action, MoveStatus, check_move_status, get_opponent, PLAYER1, PLAYER2), Node
Classes:
Node: Represents a node in the MCTS tree (imported from .node).
"""

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
)
from .node import Node


class MCTSAgent: 
    def __init__(self, iterationnumber: int = 100):
        """
        Initialize the MCTS agent with a specified number of iterations.

        Args:
            iterationnumber (int): The number of iterations for MCTS.
        """
        self.iterationnumber = iterationnumber

    def mcts_move(self, 
        board: np.ndarray,
        root_player: BoardPiece,
        saved_state: SavedState | None,
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
            while node.is_fully_expanded() and not node.is_terminal:
                node = node.best_child()
            # === EXPANSION ===
            node = self.expansion(node, player)

            # === SIMULATION ===
            result = node.result if node.is_terminal else self.simulate(node, player)

            # === BACKPROPAGATION ===
            self.backpropagate(node, result)

        # Choose the action of the most visited, for too low iterationnumber it happens that
        # the children of the current node are empty, so we need to check if there are any children
        # before trying to get the most visited child, as iterationnumber could be too low
        if not current_node.children:
            print("No children found for current_node. Returning a random valid action.")
            valid_moves = current_node.get_valid_moves()
            action = np.random.choice(valid_moves)
            return action, current_node
        most_visited = max(current_node.children.items(), key=lambda item: item[1].visits)
        action = most_visited[0]
        apply_player_action(root_node.state, action, root_player)
        saved_state = root_node
        return action, saved_state


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

        while True:
            valid_moves = [
                col
                for col in range(node_state.shape[1])
                if check_move_status(node_state, PlayerAction(col)) == MoveStatus.IS_VALID
            ]
            if not valid_moves:
                return {PLAYER1: 0, PLAYER2: 0}

            action = np.random.choice(valid_moves)
            apply_player_action(node_state, PlayerAction(action), player)

            terminal, result = Node(node_state, player).check_terminal_state()
            if terminal:
                return result

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

        # Otherwise, pick randomly as before
        action = np.random.choice(list(node.untried_actions))
        next_state = node.state.copy()
        apply_player_action(next_state, action, player)
        return action, next_state

    def __call__(self, board, player, saved_state, *args):
        return self.mcts_move(board, player, saved_state, *args)