# Connect Four Game Simulation

This project implements software agents that play the game Connect Four.  
The main focus is on applying principles of software development, especially clean code.

## Agents

- **MCTS Agent:** Plays Connect Four using Monte Carlo Tree Search.
- **Random Agent:** Makes random valid moves.
- **Human Agent:** Allows a human to play against either agent.

## Getting Started

### Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

### Installation

1. Clone this repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Simulations

To run a series of games between the MCTS agent and the Random agent:

```bash
python main.py
```

You can adjust the number of games by changing the argument in `run_mcts_vs_random()` in `main.py`.

### Playing as a Human

Uncomment one of the following lines in `main.py` to play as a human:

```python
# human_vs_agent(user_move)  # Play against yourself
# human_vs_agent(random_move, user_move, player_1="Random Agent", player_2="You")  # Play against random
# human_vs_agent(generate_move_msct, user_move, player_1="MCTS Agent", player_2="You")  # Play against MCTS
```

### Running Tests

To run all tests:

```bash
pytest tests/
```

## Project Structure

```
Game/
├── agents/
│   ├── agent_MCTS/
│   │   ├── mcts.py
│   │   └── Node.py
│   ├── agent_human_user.py
│   └── agent_random.py
├── tests/
│   ├── test_game_utils.py
│   ├── test_node.py
│   └── test_mcts.py
├── game_utils.py
└── main.py
```

## Customization

- You can modify the MCTS agent’s iteration count or logic in `agents/agent_MCTS/mcts.py`.
- Add or modify tests in the `tests/` directory.

## Code of Honour & Acknowledgements
- AI assistance (GitHub Copilot and ChatGPT) was used to help write comments, improve code style, and suggest test improvement of test covering.
- All code was written by human first and each suggestion from AI was reviewed and adapted to fit the project requirements.

## Questions or Comments?
Text me at anina.morgner@bccn-berlin.de