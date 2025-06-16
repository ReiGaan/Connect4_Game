# Connect Four Game Simulation

This project implements software agents that play the game Connect Four.  
The main focus is on applying principles of software development, especially clean code.

https://miro.com/app/board/uXjVIsyA0Qk=/

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

You will be prompted to select the game mode (e.g., Human vs Random Agent, Human vs MCTS Agent, MCTS vs Random Agent, Human vs Human).

--- 

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
│   │   ├── Node.py
|   |   └── __init__.py
│   ├── agent_random/
│   │   ├── human_user.py
│   │   └── __init__.py
│   └── agent_human_user/
│       ├── human_user.py
│       └── __init__.py
├── tests/
│   ├── test_game_utils.py
│   ├── test_node.py
│   └── test_mcts.py
├── game_utils.py
└── main.py
```

## Customization

- You can modify the MCTS agent’s iteration count or logic in `agents/agent_MCTS/mcts.py` or in the `main.py` as `args_2`
- Add or modify tests in the `tests/` directory.

## Code of Honour & Acknowledgements
- AI assistance (GitHub Copilot and ChatGPT) was used to help write comments, improve code style, and suggest test improvement of test covering.
- All code was written by human first and each suggestion from AI was reviewed and adapted to fit the project requirements.

## Questions or Comments?
Text me at anina.morgner@bccn-berlin.de
