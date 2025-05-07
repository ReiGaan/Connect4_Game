import numpy as np
from game_utils import PlayerAction, check_move_status, MoveStatus

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {} 
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.get_valid_moves()
    
    def get_valid_moves(self):
        valid_moves = [
        col for col in range(self.state.shape[1])
        if check_move_status(self.state, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        return valid_moves

    def expand(self, action, next_state):
        child_node = Node(next_state, parent=self)
        self.children[action] = child_node
        self.untried_actions.remove(action)
        return child_node
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def uct(self, node, parent_visits):
        if node.visits == 0:
            return float('inf')  # Force exploration if the node has not been visited
        win_ratio = node.wins / node.visits
        explore = np.sqrt(np.log(parent_visits) / node.visits)
        return win_ratio + explore

    def best_child(self):
        parent_visits = self.visits
        uct_scores = []

        for action, child in self.children.items():
            score = self.uct(child, parent_visits)
            uct_scores.append((score, child))  
        best_child_node = max(uct_scores, key=lambda uct_children_tuple: uct_children_tuple[0])[1] 
        
        return best_child_node