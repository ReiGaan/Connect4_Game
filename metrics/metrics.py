import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

class GameMetrics:
    """
    Tracks performance metrics for game agents including:
    - Win Rate
    - Time per Move
    - Move Accuracy

    Metrics are stored per agent in a dictionary:
    {
        'wins': int,
        'losses': int,
        'draws': int,
        'total_time': float,
        'legal_moves': int,
        'illegal_moves': int
    }
    """

    def __init__(self):
        self.agents = {}
        self.history = {}  

    def add_agent(self, agent_name: str):
        """Initializes metrics for a new agent."""
        if agent_name not in self.agents:
            self.agents[agent_name] = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_time': 0.0,
            'legal_moves': 0,
            'illegal_moves': 0
            }
            self.history[agent_name] = {
            'results': [],
            'move_times': [],
            'move_legality': []
            }
    def record_move(self, agent_name: str, time_taken: float, is_legal: bool = True):
        """
        Records a move for an agent including its duration and legality.

        Args:
            agent_name: Identifier for the agent.
            time_taken: Time taken for the move.
            is_legal: Whether the move was legal.
        """
        if agent_name not in self.agents:
            self.add_agent(agent_name)
        self.agents[agent_name]['total_time'] += time_taken
        if is_legal:
            self.agents[agent_name]['legal_moves'] += 1
        else:
            self.agents[agent_name]['illegal_moves'] += 1

        self.history[agent_name]['move_times'].append(time_taken)
        self.history[agent_name]['move_legality'].append(is_legal)

    def record_result(self, agent_name: str, result: str):
        """
        Records the outcome of a game for an agent.

        Args:
            agent_name: Identifier for the agent.
            result: One of 'win', 'loss', or 'draw'.
        """
        if agent_name not in self.agents:
            self.add_agent(agent_name)
        if result == 'win':
            self.agents[agent_name]['wins'] += 1
        elif result == 'loss':
            self.agents[agent_name]['losses'] += 1
        elif result == 'draw':
            self.agents[agent_name]['draws'] += 1

        self.history[agent_name]['results'].append(result)  

    def win_rate(self, agent_name: str) -> float:
        """
        Calculates the win rate of an agent.

        Args:
            agent_name: Identifier for the agent.

        Returns:
            Win rate as a float between 0 and 1.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
        total_games = agent['wins'] + agent['losses'] + agent['draws']
        return agent['wins'] / total_games if total_games else 0.0

    def time_per_move(self, agent_name: str) -> float:
        """
        Calculates the average time per move for an agent.

        Args:
            agent_name: Identifier for the agent.

        Returns:
            Average time per move in seconds.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
        total_moves = agent['legal_moves'] + agent.get('illegal_moves', 0)
        return agent['total_time'] / total_moves if total_moves else 0.0

    def move_accuracy(self, agent_name: str) -> float:
        """
        Calculates the proportion of legal moves made by an agent.

        Args:
            agent_name: Identifier for the agent.

        Returns:
            Accuracy as a float between 0 and 1.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
        total_moves = agent['legal_moves'] + agent.get('illegal_moves', 0)
        return agent['legal_moves'] / total_moves if total_moves else 0.0

    def plot_move_duration_distribution(self, agent_name: str):
        """
        Plots histogram of move durations for a given agent.

        Args:
            agent_name: Identifier for the agent.
        """
        import matplotlib.pyplot as plt

        if agent_name not in self.history:
            print(f"No history for agent '{agent_name}'")
            return

        move_times = self.history[agent_name]['move_times']
        if not move_times:
            print(f"No move times recorded for agent '{agent_name}'")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(move_times, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Move Duration Distribution: {agent_name}')
        plt.xlabel('Time per Move (s)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_performance_radar(self):
        """
        Plots a radar chart comparing agent performance on:
        - Win Rate
        - Move Accuracy
        - Inverse Time per Move
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.agents:
            print("No agents to compare.")
            return

        labels = ['Win Rate', 'Move Accuracy', 'Speed (1 / Time per Move)']
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for agent in self.agents:
            win_rate = self.win_rate(agent)
            accuracy = self.move_accuracy(agent)
            time = self.time_per_move(agent)
            speed = 1.0 / time if time > 0 else 0.0

            # Normalize to 0-1 scale (for fair comparison)
            values = [win_rate, accuracy, speed]
            max_vals = [1.0, 1.0, max(1.0, speed)]  # prevent divide-by-zero
            normalized = [v / m for v, m in zip(values, max_vals)]
            normalized += normalized[:1]

            ax.plot(angles, normalized, label=agent)
            ax.fill(angles, normalized, alpha=0.1)

        ax.set_title("Agent Performance Comparison (Radar Chart)", y=1.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

    


    def plot_results(self, save_path: str = None, show: bool = True):
        """
        Generates visualizations of agent performance.
        
        Args:
            save_path: Optional path to save the plot image.
            show: Whether to display the plot on screen.
        """
        if not self.agents:
            print("No metrics to plot")
            return

        agent_names = list(self.agents.keys())
        plt.figure(figsize=(14, 10))

        # Subplot 1: Game Outcomes (Grouped Bar Chart)
        plt.subplot(2, 2, 1)
        total_games = [
            self.agents[agent]['wins'] + self.agents[agent]['losses'] + self.agents[agent]['draws'] 
            for agent in agent_names
        ]
        win_percentages = [
            self.agents[agent]['wins'] / total_games[i] * 100 if total_games[i] > 0 else 0.0 
            for i, agent in enumerate(agent_names)
        ]
        loss_percentages = [
            self.agents[agent]['losses'] / total_games[i] * 100 if total_games[i] > 0 else 0.0 
            for i, agent in enumerate(agent_names)
        ]
        draw_percentages = [
            self.agents[agent]['draws'] / total_games[i] * 100 if total_games[i] > 0 else 0.0 
            for i, agent in enumerate(agent_names)
        ]

        x = np.arange(len(agent_names))
        width = 0.25

        plt.bar(x - width, win_percentages, width, label='Wins', color='#4CAF50')
        plt.bar(x, loss_percentages, width, label='Losses', color='#F44336')
        plt.bar(x + width, draw_percentages, width, label='Draws', color='#FFC107')

        plt.ylabel('Percentage (%)')
        plt.title('Game Outcomes by Agent')
        plt.xticks(x, agent_names)
        plt.legend()
        plt.ylim(0, 100)

        # Annotate percentages on bars
        for i in range(len(agent_names)):
            plt.text(x[i] - width, win_percentages[i] + 1, f'{win_percentages[i]:.1f}%', ha='center', va='bottom')
            plt.text(x[i], loss_percentages[i] + 1, f'{loss_percentages[i]:.1f}%', ha='center', va='bottom')
            plt.text(x[i] + width, draw_percentages[i] + 1, f'{draw_percentages[i]:.1f}%', ha='center', va='bottom')

        # Subplot 2: Win Rates
        plt.subplot(2, 2, 2)
        win_rates = [self.win_rate(agent) * 100 for agent in agent_names]
        bars = plt.bar(agent_names, win_rates, color='#2196F3')
        plt.ylabel('Win Rate (%)')
        plt.title('Agent Win Rates')
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', va='bottom', 
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        # Subplot 3: Move Accuracy
        plt.subplot(2, 2, 3)
        accuracies = [self.move_accuracy(agent) * 100 for agent in agent_names]
        bars = plt.bar(agent_names, accuracies, color='#9C27B0')
        plt.ylabel('Accuracy (%)')
        plt.title('Move Accuracy')
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', va='bottom', 
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        # Subplot 4: Time per Move
        plt.subplot(2, 2, 4)
        times = [self.time_per_move(agent) for agent in agent_names]
        bars = plt.bar(agent_names, times, color='#FF9800')
        plt.ylabel('Seconds')
        plt.title('Average Time per Move')
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}s', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', va='bottom', 
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        plt.tight_layout(pad=3.0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")

        if show:
            plt.show()

    def __str__(self):
        """
        Generates a readable summary report for all agents.

        Returns:
            A multi-line string summarizing metrics for each agent.
        """
        report = []
        for agent, stats in self.agents.items():
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            total_moves = stats['legal_moves'] + stats.get('illegal_moves', 0)
            report.append(f"Agent: {agent}")
            report.append(f"  Games: {total_games} (W: {stats['wins']}, L: {stats['losses']}, D: {stats['draws']})")
            report.append(f"  Win Rate: {self.win_rate(agent):.2%}")
            report.append(f"  Move Accuracy: {self.move_accuracy(agent):.2%}")
            report.append(f"  Avg Time/Move: {self.time_per_move(agent):.4f}s")
            report.append(f"  Total Moves: {total_moves} (Legal: {stats['legal_moves']}, Illegal: {stats['illegal_moves']})")
        return "\n".join(report)
