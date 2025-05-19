import cProfile
import pstats
from game_utils import initialize_game_state, PLAYER1
from agents.agent_MCTS.mcts import mcts_move

def run_mcts():
    board = initialize_game_state()
    player = PLAYER1
    saved_state = None
    # You can adjust iterationnumber for a realistic test
    mcts_move(board.copy(), player, saved_state, iterationnumber=500)

if __name__ == "__main__":
    cProfile.run('run_mcts()', 'mcts_profile.stats')
    p = pstats.Stats('mcts_profile.stats')
    p.strip_dirs().sort_stats('tottime').print_stats(20)  # Top 20 by cumulative time