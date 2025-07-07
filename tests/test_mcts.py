import numpy as np
from agents.agent_MCTS.mcts import MCTSAgent


def test_backpropagation():
    """
    Test the backpropagation function of the MCTS agent, by simulating a simple tree
    structure and verifying the updates to the visit count and win statistics.
    """
    from agents.agent_MCTS.node import Node
    from game_utils import PLAYER1, PLAYER2

    root_node = Node(state=np.zeros((6, 7)), parent=None, player=PLAYER1)
    child_node_1 = Node(state=np.zeros((6, 7)), parent=root_node, player=PLAYER2)
    child_node_2 = Node(state=np.zeros((6, 7)), parent=child_node_1, player=PLAYER1)

    root_node.visits = 0
    child_node_1.visits = 0
    child_node_2.visits = 0

    final_result = {PLAYER1: 1, PLAYER2: -1}

    agent = MCTSAgent()
    agent.backpropagate(child_node_2, final_result)

    assert root_node.visits == 1 and root_node.wins[PLAYER1] == 1


def test_mcts_never_plays_illegal_move():
    """
    Test that the MCTS agent never selects an illegal move by repeatedly provides
    a nearly full Connect Four board to the MCTS agent and checks that the move selected
    is always valid.
    """
    from game_utils import (
        BoardPiece,
        MoveStatus,
        check_move_status,
        NO_PLAYER,
        PLAYER1,
        PLAYER2,
    )

    agent = MCTSAgent(iterationnumber=10)
    num_trials = 10

    for i in range(num_trials):
        board = np.array(
            [
                [NO_PLAYER, PLAYER1, PLAYER2, NO_PLAYER, PLAYER2, PLAYER1, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1],
                [PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1],
                [NO_PLAYER, PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2],  # Make at least one move valid
            ]
        )

        player = BoardPiece(1)
        saved_state = None
        action, saved_state = agent.mcts_move(
            board.copy(),
            player,
            saved_state,
        )

        try:
            action, saved_state = agent.mcts_move(board.copy(), player, saved_state)
            move_status = check_move_status(board, action)
            assert move_status == MoveStatus.IS_VALID
        except AssertionError:
            print(f"Test failed on iteration {i}")
            print("Board before move:\n", board)
            print("Action selected:", action)
            print("Saved node state:\n", saved_state if saved_state else "None")
            print(
                "Valid moves from board:",
                [
                    c
                    for c in range(board.shape[1])
                    if check_move_status(board, c) == MoveStatus.IS_VALID
                ],
            )
            raise


def test_mcts_always_wins_against_random():
    """
    Test that the MCTS agent always wins against a random agent (baseline) in a series of Connect Four games.
    PLAYER1 uses the MCTS agent and PLAYER2 uses a random agent.
    """
    from game_utils import (
        initialize_game_state,
        PLAYER1,
        PLAYER2,
        PlayerAction,
        check_move_status,
        MoveStatus,
        apply_player_action,
    )

    def random_agent(board, player):
        valid_moves = [
            PlayerAction(col)
            for col in range(board.shape[1])
            if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        return np.random.choice(valid_moves)

    agent = MCTSAgent()
    num_games = 5
    for _ in range(num_games):
        board = initialize_game_state()
        player = PLAYER1
        saved_state = None
        winner = None

        for _ in range(42):  # Max moves in Connect Four
            if player == PLAYER1:
                action, saved_state = agent.mcts_move(
                    board.copy(),
                    player,
                    saved_state,
                )
            else:
                action = random_agent(board, player)
            apply_player_action(board, action, player)
            # Check for win
            from game_utils import connected_four

            if connected_four(board, player):
                winner = player
                break
            player = PLAYER2 if player == PLAYER1 else PLAYER1

        assert winner == PLAYER1, f"MCTS did not win, winner was {winner}"


def test_simulation_switches_players_correctly():
    """
    Tests that the simulation function correctly switches players during the simulation.
    """
    from agents.agent_MCTS.node import Node
    from game_utils import PLAYER1, initialize_game_state

    agent = MCTSAgent()
    # Set up a board where PLAYER1 can win in one move
    board = initialize_game_state()
    board[5, 2:5] = PLAYER1
    print(board)
    node = Node(state=board, player=PLAYER1)
    result = agent.simulate(node, PLAYER1)
    print(result)
    assert result[PLAYER1] == 1
