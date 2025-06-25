from .mcts import MCTSAgent
from game_utils import (
    PlayerAction, PLAYER1, PLAYER2,  BoardPiece, MoveStatus, apply_player_action, check_move_status, get_opponent
)
from .node import Node
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class HierachicalMCTSAgent(MCTSAgent):
    """An extended MCTS agent for Connect4 with heuristic-guided simulation 
    and MinMax integration.
    This agent extends the base MCTSAgent by several enhancements:
    - Uses MinMax search for critical moves when the simulation depth exceeds a threshold.
    - During expansion, prioritizes immediate winning moves and blocks opponent's immediate wins.
    - In simulations, prefers moves that create two- or three-in-a-row, and favors the center column.
    Functions:
        minmax_move(board, player) -> PlayerAction:
            Performs a MinMax search to select the best move for the current player.
        minmax(board, player, depth) -> int:
            Recursively evaluates moves using MinMax up to a specified depth.
        expand_to_next_children(player, node) -> tuple[PlayerAction, np.ndarray]:
            Expands the tree by selecting a move, prioritizing immediate wins and blocks.
        count_n_in_a_row(board, player, n) -> int:
            Counts the number of n-in-a-row occurrences for the given player.
        simulate(node, player) -> dict[BoardPiece, int]:
            Performs a heuristic-guided simulation from the given node.
        __call__(board, player, saved_state, *args):
            Returns the agent's move using MCTS.
    Attributes:
        iterationnumber (int): Number of MCTS iterations per move.
        max_depth_for_minmax (int): Depth threshold for switching to MinMax in simulation.
        max_simulation_depth (int): Maximum depth for simulation rollouts.
    """
        
    def __init__(self, iterationnumber: int = 50, max_depth_for_minmax: int = 20, max_simulation_depth: int = 40):
        super().__init__(iterationnumber)
        self.max_depth_for_minmax = max_depth_for_minmax 
        self.max_simulation_depth = max_simulation_depth

    def minmax_move(self, board: np.ndarray, player: BoardPiece, root_player: BoardPiece) -> PlayerAction:
        """
        Perform a MinMax search to find the best move for the current player.
        This function returns the action that leads to the best possible state.
        Args:
            board (np.ndarray): The current game board state.
            player (BoardPiece): The player for whom to find the best move.
            root_player (BoardPiece): The player from whose perspective the MinMax evaluation is done.
        Returns:
            PlayerAction: The best action for the player based on MinMax evaluation.
        """
        # Define the base MinMax logic (example, adjust for your game rules)
        best_score = -float('inf')
        best_move = None

        for col in range(board.shape[1]):
            if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID:
                temp_board = board.copy()
                apply_player_action(temp_board, PlayerAction(col), player)
                score = self.minmax(temp_board, get_opponent(player), root_player=root_player, depth=3, alpha=-float('inf'), beta=float('inf'))
                if score > best_score:
                    best_score = score
                    best_move = PlayerAction(col)
                    

        return best_move

    def minmax(self, board: np.ndarray, player: BoardPiece, root_player: BoardPiece, depth: int, alpha: float, beta: float) -> int:
        """
        Perform a MinMax search with a specified depth.
        This function recursively evaluates all possible moves for the given player.
        Args:
            board (np.ndarray): The current game board state.               
            player (BoardPiece): The player for whom to evaluate moves.
            depth (int): The depth to search in the tree.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.
        Returns:
            int: The score for the player at the current board state.
        """
        terminal, result = Node(board, player).check_terminal_state()
        if terminal:
            return result[root_player]  # Evaluate from root player's perspective

        if depth == 0:
            return 0  # Neutral evaluation at depth limit

        if player == root_player:
            best_score = -float('inf')
            for col in range(board.shape[1]):
                if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID:
                    temp_board = board.copy()
                    apply_player_action(temp_board, PlayerAction(col), player)
                    score = self.minmax(temp_board, get_opponent(player), root_player, depth - 1, alpha, beta)
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
            return best_score
        else:
            best_score = float('inf')
            for col in range(board.shape[1]):
                if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID:
                    temp_board = board.copy()
                    apply_player_action(temp_board, PlayerAction(col), player)
                    score = self.minmax(temp_board, get_opponent(player), root_player, depth - 1, alpha, beta)
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
            return best_score

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

    def count_n_in_a_row(self, board, player, n) -> int:
        """Count the number of n-in-a-row for the given player on the board.
        Args:  
            board (np.ndarray): The game board.
            player (BoardPiece): The player to check for.
            n (int): The number of pieces in a row to count.                    
        Returns:                        
            int: The count of n-in-a-row for the player.    
        """
        count = 0
        rows, cols = board.shape
        # Horizontal
        for r in range(rows):
            for c in range(cols - n + 1):
                if np.all(board[r, c:c+n] == player):
                    count += 1
        # Vertical
        for c in range(cols):
            for r in range(rows - n + 1):
                if np.all(board[r:r+n, c] == player):
                    count += 1
        # Diagonal /
        for r in range(n-1, rows):
            for c in range(cols - n + 1):
                if np.all([board[r-i, c+i] == player for i in range(n)]):
                    count += 1
        # Diagonal \
        for r in range(rows - n + 1):
            for c in range(cols - n + 1):
                if np.all([board[r+i, c+i] == player for i in range(n)]):
                    count += 1
        return count

    def simulate(self, node: Node, player: BoardPiece) -> dict[BoardPiece, int]:
        """
        Heuristic-guided simulation: prefer center and create two/three-in-a-row.
        Args: 
            node (Node): The node to simulate from.
            player (BoardPiece): The player to simulate for.
        Returns:
            dict[BoardPiece, int]: The result of the simulation for each player.
        """
        root_player = player
        node_state = node.state.copy()
        depth = 0 
        self.max_simulation_depth = 20
        while depth < self.max_simulation_depth:
            valid_moves = [
            col
            for col in range(node_state.shape[1])
            if check_move_status(node_state, PlayerAction(col)) == MoveStatus.IS_VALID
            ]
            if not valid_moves:
                return {PLAYER1: 0, PLAYER2: 0}

            # Check for immediate win
            for action in valid_moves:
                temp_state = node_state.copy()
                apply_player_action(temp_state, PlayerAction(action), player)
                terminal, result = Node(temp_state, player).check_terminal_state()
                if terminal and result[player] > 0:
                    apply_player_action(node_state, PlayerAction(action), player)
                    return result

            if depth >= self.max_depth_for_minmax:
            # Use MinMax for critical moves when the threshold is reached
                action = self.minmax_move(node_state, player, root_player=root_player)
            else:

                # Prefer moves that create three-in-a-row, then two-in-a-row, then center, then random
                best_score = -1
                best_moves = []
                for action in valid_moves:
                    temp_state = node_state.copy()
                    apply_player_action(temp_state, PlayerAction(action), player)
                    score = (
                        9 * self.count_n_in_a_row(temp_state, player, 3) +
                        3 * self.count_n_in_a_row(temp_state, player, 2)
                    )
                    # Prefer center column slightly
                    if action == node_state.shape[1] // 2:
                        score += 1
                    if score > best_score:
                        best_score = score
                        best_moves = [action]
                    elif score == best_score:
                        best_moves.append(action)
                action = np.random.choice(best_moves)
                apply_player_action(node_state, PlayerAction(action), player)

                terminal, result = Node(node_state, player).check_terminal_state()
                if terminal:
                    return result

            player = get_opponent(player)
            depth += 1

        # If the loop ends without reaching a terminal state or max depth
        print(f"Maximum simulation depth reached without terminal state.")
        return {PLAYER1: 0, PLAYER2: 0}

    def __call__(self, board, player, saved_state, *args):
        return self.mcts_move(board, player, saved_state, *args)