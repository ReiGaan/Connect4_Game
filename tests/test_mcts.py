import numpy as np

def test_backpropagation():
    """
    Test the backpropagation function of the MCTS agent, by simulating a simple tree
    structure and verifying the updates to the visit count and win statistics.
    """
    from agents.agent_MCTS.node import Node
    from agents.agent_MCTS.mcts import backpropagate
    from game_utils import PLAYER1, PLAYER2

    root_node = Node(state=np.zeros((6, 7)), parent=None, player=PLAYER1)
    child_node_1 = Node(state=np.zeros((6, 7)), parent=root_node, player=PLAYER2)
    child_node_2 = Node(state=np.zeros((6, 7)), parent=child_node_1, player=PLAYER1)

    root_node.visits = 0
    child_node_1.visits = 0
    child_node_2.visits = 0

    final_result = {PLAYER1: 1, PLAYER2: -1}

    backpropagate(child_node_2, final_result)

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
    from agents.agent_MCTS.mcts import mcts_move

    num_trials = 10

    for i in range(num_trials):
        board = np.array(
            [
                [NO_PLAYER, PLAYER1, PLAYER2, NO_PLAYER, PLAYER2, PLAYER1, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER1],
                [PLAYER2, PLAYER1, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2, PLAYER2],
                [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER1, PLAYER1],
                [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER1, PLAYER1, PLAYER2],
            ]
        )

        player = BoardPiece(1)
        saved_state = None
        action, saved_state = mcts_move(
            board.copy(), player, saved_state, iterationnumber=10
        )

        try:
            action, saved_state = mcts_move(board.copy(), player, saved_state)
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
    from agents.agent_MCTS.mcts import mcts_move
    import numpy as np

    def random_agent(board, player):
        valid_moves = [
            PlayerAction(col)
            for col in range(board.shape[1])
            if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        return np.random.choice(valid_moves)

    num_games = 5
    for _ in range(num_games):
        board = initialize_game_state()
        player = PLAYER1
        saved_state = None
        winner = None

        for _ in range(42):  # Max moves in Connect Four
            if player == PLAYER1:
                action, saved_state = mcts_move(
                    board.copy(), player, saved_state, iterationnumber=1000
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
    from agents.agent_MCTS.mcts import simulate
    from game_utils import PLAYER1, initialize_game_state

    # Set up a board where PLAYER1 can win in one move
    board = initialize_game_state()
    board[5, 2:5] = PLAYER1
    print(board)
    node = Node(state=board, player=PLAYER1)
    result = simulate(node, PLAYER1)
    print(result)
    assert result[PLAYER1] == 1

def test_mcts_finds_winning_move_horizontal():
    """
    Tests that MCTS selects the winning move (horizontal) when there is one available.
    """
    from game_utils import initialize_game_state, PLAYER1, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 2:5] = PLAYER1
    print(board)
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state, iterationnumber=5000)
    print(action)
    assert action == PlayerAction(1) or action == PlayerAction(5)

def test_mcts_finds_diagonal_win():
    """
    Tests that MCTS finds a diagonal win.
    """
    from game_utils import (
        initialize_game_state,
        NO_PLAYER,
        PLAYER1,
        PLAYER2,
        PlayerAction,
    )
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board = np.array(
        [
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, PLAYER1, PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER],
            [PLAYER1, PLAYER2, PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER],
        ]
    )
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(3)

def test_mcts_finds_diagonal_backslash_win():
    """
    Tests that MCTS finds a diagonal win in the "\" direction.
    """
    from game_utils import (
        initialize_game_state,
        NO_PLAYER,
        PLAYER1,
        PLAYER2,
        PlayerAction,
    )
    from agents.agent_MCTS.mcts import mcts_move

    board = np.array(
        [
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER2, PLAYER2, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER],
        ]
    )

    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(3)

def test_mcts_finds_winning_move_other_player_starts():
    """
    Tests that MCTS selects the winning move when PLAYER2 is the agent and has a win available.
    """
    from game_utils import initialize_game_state, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 2:5] = PLAYER2
    player = PLAYER2
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(1) or action == PlayerAction(5)

def test_mcts_multiple_winning_moves():
    """
    Tests that MCTS picks any of several available winning moves.
    """
    from game_utils import initialize_game_state, PLAYER1, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 1:3] = PLAYER1
    board[5, 4:6] = PLAYER1
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action in [PlayerAction(0), PlayerAction(3), PlayerAction(6)]

def test_mcts_defends_move_vertical():
    """
    Tests that MCTS defends the winning move of the opponent when there is one available
    and there is a way to defend it.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[3:6, 2] = PLAYER2
    print(board)
    player = PLAYER1
    saved_state = None
    action, _ = mcts_move(board.copy(), player, saved_state)
    print(action)

    assert action == PlayerAction(2)

def test_mcts_defends_diagonal():
    from game_utils import (
        initialize_game_state,
        PLAYER1,
        PLAYER2,
        NO_PLAYER,
        PlayerAction,
    )
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board = np.array(
        [
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
                NO_PLAYER,
            ],
            [NO_PLAYER, NO_PLAYER, PLAYER2, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, PLAYER2, PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER2, PLAYER1, NO_PLAYER, PLAYER1],
        ]
    )

    print(board)
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    print(action)
    assert action == PlayerAction(3)

def test_mcts_defends_horizontal():
    """
    Tests that MCTS blocks a horizontal win by the opponent.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[5, 2:5] = PLAYER2
    board[5, 1] = PLAYER1
    player = PLAYER1
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(5)

def test_mcts_defends_move_other_player_starts():
    """
    Tests that MCTS (as PLAYER2) defends the winning move of PLAYER1 when there is one available.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    from agents.agent_MCTS.mcts import mcts_move

    board = initialize_game_state()
    board[3:6, 2] = PLAYER1
    board[5, 3:5] = PLAYER2

    player = PLAYER2
    saved_state = None

    action, _ = mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(2)
