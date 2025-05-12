import numpy as np
from game_utils import PlayerAction, check_move_status, MoveStatus

class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS).

    Attributes:
        state (np.ndarray): The game state associated with this node.
        parent (Node | None): The parent node in the search tree.
        children (dict): A dictionary mapping actions to child nodes.
        visits (int): Number of times this node has been visited.
        wins (float): Simulation result scores for this node winning.
        untried_actions (list): List of actions not yet tried from this state.
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {} 
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.get_valid_moves()
    
    def get_valid_moves(self):
        """
        Compute the list of valid moves from the current Node.

        Returns:
            list[int]: A list of column indices that are valid moves.
        """
        
        valid_moves = [
        col for col in range(self.state.shape[1])
        if check_move_status(self.state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        return valid_moves

    def expand(self, action, next_state):
        """
        Expand the current node by adding a child node for the given action.

        Args:
            action (PlayerAction): The action that leads to the child state.
            next_state (np.ndarray): The resulting state from applying the action.

        Returns:
            child_node (Node): The newly created child node.
        """
        child_node = Node(next_state, parent=self)
        self.children[action] = child_node
        self.untried_actions.remove(action)
        return child_node
    
    def is_fully_expanded(self):
        """
        Check whether all possible actions from this node have been tried.

        Returns:
            bool: True if no untried actions remain, False otherwise.
        """
        return len(self.untried_actions) == 0

    def uct(self, node, parent_visits):
        """
        Calculate the Upper Confidence Bound (UCT) score for a child node.

        Args:
            node (Node): The child node to evaluate.
            parent_visits (int): The number of visits to the parent node.

        Returns:
            float: The UCT score for the node.
        """
        if node.visits == 0:
            return float('inf')  # Force exploration if the node has not been visited
        win_ratio = node.wins / node.visits
        explore = np.sqrt(np.log(parent_visits) / node.visits)
        return win_ratio + explore

    def best_child(self):
        """
        Select the child node with the highest UCT score.

        Returns:
            best_child (Node): The child node with the highest UCT value.
        """
        parent_visits = self.visits
        uct_scores = []

        for action, child in self.children.items():
            score = self.uct(child, parent_visits)
            uct_scores.append((score, child))  
        best_child_node = max(uct_scores, key=lambda uct_children_tuple: uct_children_tuple[0])[1] 
        
        return best_child_node
    
    def refresh_children(self):
        """
        Refresh the node's valid children and untried actions to reflect current state.
        Removes children whose moves are no longer valid.
        """
        valid_moves = self.get_valid_moves()
        for action in list(self.children.keys()):
            if action not in valid_moves:
                del self.children[action]
        self.untried_actions = list(set(valid_moves) - set(self.children.keys()))
