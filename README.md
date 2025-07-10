# Connect Four Game Simulation

This project implements software agents that play the game Connect Four.  
The main focus is on applying principles of software development, especially clean code.

https://miro.com/app/board/uXjVIsyA0Qk=/


## Agents

- **MCTS Agent:** Classic Monte Carlo Tree Search agent.
- **Hierarchical MCTS Agent:** MCTS with with heuristics (immediate win/lose detection, two/three-in-a-row preference).
- **AlphaZero MCTS Agent:** MCTS guided by a neural network (AlphaZero-style).
- **Random Agent:** Selects random valid moves.
- **Human Agent:** Allows a human to play via console input.

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
   PyTorch will automatically use CUDA or Apple's MPS backend if available.

### Running Simulations

To run a series of games between the MCTS agent and the Random agent:

```bash
python main.py
```

You will be prompted to select the game mode (e.g., Human vs Random Agent, Human vs MCTS Agent, ...).

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
│   │   ├── improved_mcts.py
│   │   ├── hierarchical_mcts.py
│   │   ├── alphazero_mcts.py
│   │   ├── node.py
│   │   └── __init__.py
│   ├── agent_random/
│   │   └── __init__.py
│   └── agent_human_user/
│       ├── human_user.py
│       └── __init__.py
├── alphazero/
│   ├── network.py
│   ├── inference.py 
│   ├── model.pt
│   └── train_dummy_data.py
├── profiling/
│   ├── mcts_profile.stats 
│   ├── profile_alphanet.py
│   ├── profile_gpu.py
│   ├── profile_hirachical_mcts.py
│   ├── profile_mcts.py
│   └── ProfilingReport.py
├── metrics/
│   ├── metrics.py
│   └── __init__.py
├── tests/
│   ├── test_game_utils.py
│   ├── test_node.py
│   ├── test_mcts.py
│   ├── test_hierarchical_mcts.py
│   ├── test_metrics.py
│   └── test_training.py
├── game_utils.py
├── main.py
└── requirements.txt
└── train_alphazero.py
```

## Code of Honour & Acknowledgements
- AI assistance (GitHub Copilot and ChatGPT) was used to help write comments, improve code style, and suggest test improvement of test covering.
- All code was written by human first and each suggestion from AI was reviewed and adapted to fit the project requirements.

