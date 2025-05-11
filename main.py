from typing import Callable
import time
from game_utils import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState, MoveStatus, GenMove
from game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, check_move_status
from agents.agent_human_user import user_move
from agents.agent_random import generate_move as random_move
from agents.agent_MCTS.mcts import mcts_move as generate_move_msct

def  human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):

    players = (PLAYER1, PLAYER2)
    results = []
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                    player, saved_state[player], *args
                )
                print(f'Move time: {time.time() - t0:.3f}s')

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    print(f'Move {action} is invalid: {move_status.value}')
                    print(f'{player_name} lost by making an illegal move.')
                    playing = False
                    results.append('Error')
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print('Game ended in draw')
                        results.append('Draw') 
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                        playing = False
                        results.append(PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT)
                        break
    return results

def run_mcts_vs_random(num_games: int = 100):
    mcts_wins = 0
    random_wins = 0
    draws = 0
    errors = 0
    
    for _ in range(num_games):
        print(f"Game {_ + 1}/{num_games}")
          # Play one game between MCTS Agent and Random Agent
        results = human_vs_agent(generate_move_1=generate_move_msct, 
                                generate_move_2=random_move, 
                                player_1="MCTS Agent", 
                                player_2="Random Agent")
        for result in results:
            if result == PLAYER1_PRINT:
                mcts_wins += 1  
            elif result == PLAYER2_PRINT:
                random_wins += 1 
            elif result == 'Draw':
                draws += 1  
            elif result == 'Error':
                errors += 1
    print(f"\nResults after {num_games} games:")
    print(f"MCTS Agent wins: {mcts_wins}")  
    print(f"Random Agent wins: {random_wins}")
    print(f"Draws: {draws}")
    print(f"Errors: {errors}")    

if __name__ == "__main__":
    # Run the MCTS vs Random simulation for 10 games
    run_mcts_vs_random(10)

#if __name__ == "__main__":
    #human_vs_agent(user_move) play against myself
    #human_vs_agent(random_move, user_move, player_1="Random Agent", player_2="Anina") #play against random
    #human_vs_agent(generate_move_msct, user_move, player_1="MCTS Agent", player_2="Anina") #play against random
    #human_vs_agent(generate_move_1=generate_move_msct, generate_move_2=random_move, player_1="MCTS Agent", player_2="Random Agent") #play against random
    