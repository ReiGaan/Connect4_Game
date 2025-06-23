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

        plt.subplot(2, 2, 1)
        for agent in agent_names:
            wins = self.agents[agent]['wins']
            losses = self.agents[agent]['losses']
            draws = self.agents[agent]['draws']
            total = wins + losses + draws
            if total > 0:
                sizes = [wins, losses, draws]
                labels = ['Wins', 'Losses', 'Draws']
                colors = ['#4CAF50', '#F44336', '#FFC107']
                explode = (0.1, 0, 0)
                wedges, texts, autotexts = plt.pie(
                    sizes, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=140, explode=explode,
                    textprops=dict(color="white")
                )
                plt.legend(wedges, labels, title="Outcome", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                plt.title(f'{agent} Results Distribution')
                break

        plt.subplot(2, 2, 2)
        win_rates = [self.win_rate(agent) * 100 for agent in agent_names]
        bars = plt.bar(agent_names, win_rates, color='#2196F3')
        plt.ylabel('Win Rate (%)')
        plt.title('Agent Win Rates')
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.subplot(2, 2, 3)
        accuracies = [self.move_accuracy(agent) * 100 for agent in agent_names]
        bars = plt.bar(agent_names, accuracies, color='#9C27B0')
        plt.ylabel('Accuracy (%)')
        plt.title('Move Accuracy')
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.subplot(2, 2, 4)
        times = [self.time_per_move(agent) for agent in agent_names]
        bars = plt.bar(agent_names, times, color='#FF9800')
        plt.ylabel('Seconds')
        plt.title('Average Time per Move')
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

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
