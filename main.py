from typing import Callable
import time
from game_utils import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState, MoveStatus, GenMove
from game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, check_move_status
from agents.agent_human_user import user_move
from agents.agent_random import generate_move as random_move
from metrics.metrics import GameMetrics
from agents.agent_MCTS.mcts import MCTSAgent
from agents.agent_MCTS.hierachical_mcts import HierachicalMCTSAgent
from agents.agent_MCTS.alphazero_mcts import AlphazeroMCTSAgent
from agents.alphazero.network import Connect4Net
from agents.alphazero.inference import policy_value
import torch 


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
    metrics: GameMetrics = None,
    verbose: bool = True  # Added verbose parameter to control output
) -> tuple:
    """Run a game between two agents with integrated metrics tracking"""
    if metrics is None:
        metrics = GameMetrics()
    
    players = (PLAYER1, PLAYER2)
    results = []
    for play_first in (1, -1):
        if play_first == 1:
            inits = (init_1, init_2)
            gen_moves = (generate_move_1, generate_move_2)
            player_names = (player_1, player_2)
            gen_args = (args_1, args_2)
        else:
            inits = (init_2, init_1)
            gen_moves = (generate_move_2, generate_move_1)
            player_names = (player_2, player_1)
            gen_args = (args_2, args_1)

        for init, player in zip(inits, players):
            init(initialize_game_state(), player)

        saved_state: dict = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        player_name_map = {players[0]: player_names[0], players[1]: player_names[1]}

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                if verbose:  # Only show board if verbose mode
                    print(pretty_print_board(board))
                    print(
                        f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                    )

                action, saved_state[player] = gen_move(
                    board.copy(),  
                    player, saved_state[player], player_name, metrics, *args
                )
                elapsed = time.time() - t0
                if verbose:  # Only show move time if verbose
                    print(f'Move time: {elapsed:.3f}s')
                
                # Record move metrics
                move_status = check_move_status(board, action)
                is_legal = move_status == MoveStatus.IS_VALID
                metrics.record_move(player_name, elapsed, is_legal)
                
                if move_status != MoveStatus.IS_VALID:
                    if verbose:  # Only show error if verbose
                        print(f'Move {action} is invalid: {move_status.value}')
                        print(f'{player_name} lost by making an illegal move.')
                    
                    # Record results for illegal move
                    metrics.record_result(player_name, 'loss')
                    opponent = PLAYER2 if player == PLAYER1 else PLAYER1
                    metrics.record_result(player_name_map[opponent], 'win')
                    
                    playing = False
                    results.append('Error')
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    if verbose:  # Only show final board if verbose
                        print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        if verbose:  # Only show result if verbose
                            print('Game ended in draw')
                        # Record draw for both players
                        for name in player_name_map.values():
                            metrics.record_result(name, 'draw')
                        results.append('Draw')
                    else:
                        winner_name = player_name
                        loser = PLAYER2 if player == PLAYER1 else PLAYER1
                        loser_name = player_name_map[loser]
                        if verbose:  # Only show result if verbose
                            print(f'{winner_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}')
                        
                        # Record win/loss results
                        metrics.record_result(winner_name, 'win')
                        metrics.record_result(loser_name, 'loss')
                        
                        playing = False
                        results.append(PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT)
                    break
    return results, metrics

def run_mcts_vs_random(num_games: int = 100):
    """Run multiple games between MCTS and Random agents with metrics tracking"""
    total_metrics = GameMetrics()
    
    mcts_wins_started = 0
    mcts_wins_not_started = 0  
    random_wins_started = 0
    random_wins_not_started = 0
    draws = 0
    errors = 0
    total = []
    for i in range(num_games):
        print(f"Game {i + 1}/{num_games}")
       
        # Show board for first game, then just progress
        verbose = (i == 0)
        results = human_vs_agent(generate_move_1=MCTSAgent(100), 
                                generate_move_2=random_move, 
                                player_1="MCTS Agent", 
                                player_2="Random Agent",
                                metrics=total_metrics,
                                verbose=verbose)
        total.append(results)
        if results[0] == PLAYER1_PRINT:
            mcts_wins_started += 1
        elif results[0] == PLAYER2_PRINT:
            random_wins_not_started += 1
        elif results[0] == 'Draw':
            draws += 1
        elif results[0] == 'Error':
            errors += 1

        if results[1] == PLAYER1_PRINT:
            random_wins_started += 1
        elif results[1] == PLAYER2_PRINT:
            mcts_wins_not_started += 1
        elif results[1] == 'Draw':
            draws += 1
        elif results[1] == 'Error':
            errors += 1
    
    # Print detailed results
    print(f"\nResults after {num_games} games:")
    print(f"MCTS Agent wins when starting: {mcts_wins_started}")
    print(f"MCTS Agent wins when not starting: {mcts_wins_not_started}")
    print(f"Random Agent wins when starting: {random_wins_started}")
    print(f"Random Agent wins when not starting: {random_wins_not_started}")
    print(f"Draws: {draws}")
    print(f"Errors: {errors}")
    
    # Print comprehensive metrics
    print("\nPerformance Metrics:")
    print(total_metrics)
    return total_metrics

def run_alphazero_vs_random(num_games: int, alpha_iterations=100):
    """Run multiple games between AlphaZero and Random agents"""
    total_metrics = GameMetrics()
    
    # Initialize agent
    model = Connect4Net()
    checkpoint = torch.load("checkpoints/iteration_20.pt", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    alpha_agent = AlphazeroMCTSAgent(
        policy_value=lambda state: policy_value(state, model),
        iterationnumber=alpha_iterations
    )
    
    # Track results
    alpha_wins_started = 0
    alpha_wins_not_started = 0
    random_wins_started = 0
    random_wins_not_started = 0
    draws = 0
    errors = 0
    
    for game_idx in range(num_games):
        print(f"\nGame {game_idx+1}/{num_games}")
        
        # Show board for first game only
        verbose = (game_idx == 0)
        
        # Alternate who starts
        if game_idx % 2 == 0:
            # AlphaZero starts as Player 1
            player1 = "AlphaZero Agent"
            player2 = "Random Agent"
            agent1 = alpha_agent
            agent2 = random_move
        else:
            # Random starts as Player 1
            player1 = "Random Agent"
            player2 = "AlphaZero Agent"
            agent1 = random_move
            agent2 = alpha_agent
        
        # Run the game
        results, _ = human_vs_agent(
            agent1,
            agent2,
            player_1=player1,
            player_2=player2,
            metrics=total_metrics,
            verbose=verbose
        )
        
        # Track results based on who started
        if game_idx % 2 == 0:  # AlphaZero started
            if results[0] == PLAYER1_PRINT:
                alpha_wins_started += 1
            elif results[0] == PLAYER2_PRINT:
                random_wins_not_started += 1
            elif results[0] == 'Draw':
                draws += 1
            elif results[0] == 'Error':
                errors += 1
        else:  # Random started
            if results[0] == PLAYER1_PRINT:
                random_wins_started += 1
            elif results[0] == PLAYER2_PRINT:
                alpha_wins_not_started += 1
            elif results[0] == 'Draw':
                draws += 1
            elif results[0] == 'Error':
                errors += 1
    
    # Print summary
    print(f"\nResults after {num_games} games:")
    print(f"AlphaZero wins when starting: {alpha_wins_started}")
    print(f"AlphaZero wins when not starting: {alpha_wins_not_started}")
    print(f"Total AlphaZero wins: {alpha_wins_started + alpha_wins_not_started}")
    print(f"Random wins when starting: {random_wins_started}")
    print(f"Random wins when not starting: {random_wins_not_started}")
    print(f"Total Random wins: {random_wins_started + random_wins_not_started}")
    print(f"Draws: {draws}")
    print(f"Errors: {errors}")
    
    return total_metrics

if __name__ == "__main__":
    print("Connect Four Game")
    print("Choose game mode:")
    print("1: User vs Random Agent")
    print("2: User vs MCTS Agent")
    print("3: MCTS Agent vs Random Agent (performance test)")
    print("4: Human vs Human (2 players)")
    print("5: MCTS Agent vs Hierachical MCTS Agent")
    print("6: Hierachical MCTS Agent vs random Agent (baseline test)")
    print("7: AlphaZero Agent vs Random Agent")
    mode = input("Enter number: ").strip()
    metrics = GameMetrics()

    if mode == "1":
        _, metrics = human_vs_agent(
            user_move,
            random_move,
            player_1="You",
            player_2="Random Agent",
            metrics=metrics
        )
    elif mode == "2":
        _, metrics = human_vs_agent(
            user_move,
            MCTSAgent(100),
            player_1="You",
            player_2="MCTS Agent",
            args_2=(100,),
            metrics=metrics
        )
    elif mode == "3":
        num_games = int(input("How many games? "))
        metrics = run_mcts_vs_random(num_games)
    elif mode == "4":
        _, metrics = human_vs_agent(
            user_move,
            user_move,
            player_1="Player 1",
            player_2="Player 2",
            metrics=metrics
        )
    elif mode == "5":
        num_games = int(input("How many games? "))
        for i in range(num_games):
            print(f"Game {i + 1}/{num_games}")
            verbose = (i == 0)  # Show board for first game only
            human_vs_agent(
                MCTSAgent(100),  
                HierachicalMCTSAgent(iterationnumber=50),  
                player_1="MCTS Agent",
                player_2="Hierachical MCTS Agent",
                metrics=metrics,
                verbose=verbose
            )
    elif mode == "6":
        # Show full game for this single match
        human_vs_agent(
            HierachicalMCTSAgent(25),  
            random_move,  
            player_1="Hierachical MCTS Agent",
            player_2="Random Agent",
            metrics=metrics,
            verbose=True
        )
    elif mode == "7":
        num_games = int(input("How many games to play? "))
        metrics = run_alphazero_vs_random(num_games)
    else:
        print("Invalid selection.")
        exit()

    # Always show metrics summary
    print("\nFinal Performance Metrics:")
    print(metrics)

    # Ask user if they want to see visualizations
    plot_choice = input("\nWould you like to see performance visualizations? (y/n): ").strip().lower()
    if plot_choice == 'y':
        metrics.plot_results()
        for agent in metrics.agents:
            metrics.plot_move_duration_distribution(agent)
    metrics.plot_performance_radar()
    
