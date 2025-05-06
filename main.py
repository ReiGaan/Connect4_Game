import game_utils as game

board = game.initialize_game_state()
board[5,1] = game.PLAYER2
board[4,2] = game.PLAYER2
board[3,3] = game.PLAYER2
board[2,4] = game.PLAYER2
print(board)
print(game.test_connect_diagonal(board, game.PLAYER2))