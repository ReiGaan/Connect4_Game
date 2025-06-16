import numpy as np
from agents.agent_MCTS.improved_mcts import ImprovedMCTSAgent
from game_utils import PLAYER1, PLAYER2, PlayerAction, initialize_game_state
from agents.agent_MCTS.mcts import MCTSAgent

def test_minmax_move_returns_valid_action_on_empty_board():
    agent = ImprovedMCTSAgent()
    empty_board = np.zeros((6, 7), dtype=int)
    action = agent.minmax_move(empty_board, PLAYER1)
    assert isinstance(action, PlayerAction)
    assert 0 <= int(action) < empty_board.shape[1]

def test_minmax_move_avoids_full_columns():
    agent = ImprovedMCTSAgent()
    board = np.zeros((6, 7), dtype=int)
    board[:, 0] = PLAYER1 
    action = agent.minmax_move(board, PLAYER1)
    assert int(action) != 0  # Should not pick full column

def test_minmax_move_returns_none_if_no_valid_moves():
    agent = ImprovedMCTSAgent()
    board = np.ones((6, 7), dtype=int)
    action = agent.minmax_move(board, PLAYER1)
    assert action is None

def test_minmax_move_returns_action_for_almost_full_board():
    agent = ImprovedMCTSAgent()
    board = np.ones((6, 7), dtype=int)
    board[0, 6] = 0  # Only one move left
    action = agent.minmax_move(board, PLAYER1)
    assert int(action) == 6

def test_mcts_agent_returns_valid_move():
    empty_board = initialize_game_state()
    agent = MCTSAgent(iterationnumber=10)
    action, _ = agent.mcts_move(empty_board, PLAYER1, None)
    assert 0 <= action < empty_board.shape[1]

def test_improved_mcts_agent_returns_valid_move():
    empty_board = initialize_game_state()
    agent = ImprovedMCTSAgent(iterationnumber=10)
    action, _ = agent.mcts_move(empty_board, PLAYER2, None)
    assert 0 <= action < empty_board.shape[1]

def test_agents_do_not_crash_on_full_column():
    empty_board = initialize_game_state()
    agent = MCTSAgent(iterationnumber=5)
    # Fill up column 0
    for _ in range(empty_board.shape[0]):
        empty_board[_, 0] = PLAYER1
    action, _ = agent.mcts_move(empty_board, PLAYER1, None)
    assert action != 0  

def test_improved_agent_prefers_winning_move():
    empty_board = initialize_game_state()
    agent = ImprovedMCTSAgent(iterationnumber=5)
    # Set up a board where PLAYER2 can win in column 3
    empty_board[5, 3] = PLAYER2
    empty_board[4, 3] = PLAYER2
    empty_board[3, 3] = PLAYER2
    action, _ = agent.mcts_move(empty_board, PLAYER2, None)
    assert action == 3  

def test_mcts_finds_winning_move_horizontal():
    """
    Tests that MCTS selects the winning move (horizontal) when there is one available.
    """
    from game_utils import initialize_game_state, PLAYER1, PlayerAction
    agent = ImprovedMCTSAgent(iterationnumber=5000)
    board = initialize_game_state()
    board[5, 2:5] = PLAYER1
    print(board)
    player = PLAYER1
    saved_state = None

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
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

    agent = ImprovedMCTSAgent()
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

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
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

    agent = ImprovedMCTSAgent()
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

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(3)

def test_mcts_finds_winning_move_other_player_starts():
    """
    Tests that MCTS selects the winning move when PLAYER2 is the agent and has a win available.
    """
    from game_utils import initialize_game_state, PLAYER2, PlayerAction

    agent = ImprovedMCTSAgent()
    board = initialize_game_state()
    board[5, 2:5] = PLAYER2
    player = PLAYER2
    saved_state = None

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(1) or action == PlayerAction(5)

def test_mcts_multiple_winning_moves():
    """
    Tests that MCTS picks any of several available winning moves.
    """
    from game_utils import initialize_game_state, PLAYER1, PlayerAction
    
    agent = ImprovedMCTSAgent()
    
    board = initialize_game_state()
    board[5, 1:3] = PLAYER1
    board[5, 4:6] = PLAYER1
    player = PLAYER1
    saved_state = None

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
    assert action in [PlayerAction(0), PlayerAction(3), PlayerAction(6)]

def test_mcts_defends_move_vertical():
    """
    Tests that MCTS defends the winning move of the opponent when there is one available
    and there is a way to defend it.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    
    agent = ImprovedMCTSAgent()
    board = initialize_game_state()
    board[3:6, 2] = PLAYER2
    print(board)
    player = PLAYER1
    saved_state = None
    action, _ = agent.mcts_move(board.copy(), player, saved_state)
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
    
    agent = ImprovedMCTSAgent()

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

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
    print(action)
    assert action == PlayerAction(3)

def test_mcts_defends_horizontal():
    """
    Tests that MCTS blocks a horizontal win by the opponent.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction
    
    agent = ImprovedMCTSAgent()
    board = initialize_game_state()
    board[5, 2:5] = PLAYER2
    board[5, 1] = PLAYER1
    player = PLAYER1
    saved_state = None

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(5)

def test_mcts_defends_move_other_player_starts():
    """
    Tests that MCTS (as PLAYER2) defends the winning move of PLAYER1 when there is one available.
    """
    from game_utils import initialize_game_state, PLAYER1, PLAYER2, PlayerAction

    agent = ImprovedMCTSAgent()
    board = initialize_game_state()
    board[3:6, 2] = PLAYER1
    board[5, 3:5] = PLAYER2

    player = PLAYER2
    saved_state = None

    action, _ = agent.mcts_move(board.copy(), player, saved_state)
    assert action == PlayerAction(2)
