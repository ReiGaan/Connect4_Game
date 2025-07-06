import cProfile
import pstats
import sys
from memory_profiler import profile
from agents.agent_MCTS.mcts import MCTSAgent
from game_utils import initialize_game_state, PLAYER1

# Redirect memory profiling output to a file
memory_log = open('memory_profile_mcts.log', 'w')
sys.stdout = memory_log  # Redirect sys.stdout directly to the file

@profile
def memory_profiled_mcts_move():
    board = initialize_game_state()
    player = PLAYER1
    saved_state = None
    agent = MCTSAgent(iterationnumber=100)
    action, saved_state = agent.mcts_move(board.copy(), player, saved_state, player_name="Player1")

def cpu_profiled_mcts_move():
    board = initialize_game_state()
    player = PLAYER1
    saved_state = None
    agent = MCTSAgent(iterationnumber=100)

    profiler = cProfile.Profile()
    profiler.enable()
    action, saved_state = agent.mcts_move(board.copy(), player, saved_state, player_name="Player1")
    profiler.disable()

    # Save CPU profiling results to a file
    with open('cpu_profile_mcts.log', 'w') as cpu_log:
        stats = pstats.Stats(profiler, stream=cpu_log)
        stats.strip_dirs().sort_stats('tottime').print_stats(20)

if __name__ == "__main__":
    # Run memory profiling
    memory_profiled_mcts_move()

    # Close memory log
    memory_log.close()

    # Restore stdout for CPU profiling
    sys.stdout = sys.__stdout__

    # Run CPU profiling
    cpu_profiled_mcts_move()