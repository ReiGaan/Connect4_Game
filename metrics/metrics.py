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
    
    Metrics are stored per agent in a dictionary structure:
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
        """Initialize tracking for a new agent"""
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
        """Record move statistics for an agent"""
        if agent_name not in self.agents:
            self.add_agent(agent_name)
        self.agents[agent_name]['total_time'] += time_taken
        if is_legal:
            self.agents[agent_name]['legal_moves'] += 1
        else:
            self.agents[agent_name]['illegal_moves'] += 1
    
    def record_result(self, agent_name: str, result: str):
        """Record game result for an agent (win/loss/draw)"""
        if agent_name not in self.agents:
            self.add_agent(agent_name)
        if result == 'win':
            self.agents[agent_name]['wins'] += 1
        elif result == 'loss':
            self.agents[agent_name]['losses'] += 1
        elif result == 'draw':
            self.agents[agent_name]['draws'] += 1
    
    def win_rate(self, agent_name: str) -> float:
        """Calculate win rate for specified agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
        total_games = agent['wins'] + agent['losses'] + agent['draws']
        return agent['wins'] / total_games if total_games else 0.0
    
    def time_per_move(self, agent_name: str) -> float:
        """Calculate average time per move for specified agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
        total_moves = agent['legal_moves'] + agent.get('illegal_moves', 0)
        return agent['total_time'] / total_moves if total_moves else 0.0
    
    def move_accuracy(self, agent_name: str) -> float:
        """Calculate move accuracy for specified agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
        total_moves = agent['legal_moves'] + agent.get('illegal_moves', 0)
        return agent['legal_moves'] / total_moves if total_moves else 0.0
    
    def plot_results(self, save_path: str = None, show: bool = True):
        """Visualize agent performance metrics with plots"""
        if not self.agents:
            print("No metrics to plot")
            return
            
        agent_names = list(self.agents.keys())
        
        # Create a 2x2 grid of plots
        plt.figure(figsize=(14, 10))
        
        # Win/Loss/Draw Distribution (Pie Chart)
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
                
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=140, explode=explode)
                plt.title(f'{agent} Results Distribution')
                break  # Only show first agent for clarity
        
        # Win Rate Comparison (Bar Chart)
        plt.subplot(2, 2, 2)
        win_rates = [self.win_rate(agent) * 100 for agent in agent_names]
        bars = plt.bar(agent_names, win_rates, color='#2196F3')
        plt.ylabel('Win Rate (%)')
        plt.title('Agent Win Rates')
        plt.ylim(0, 100)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        # Move Accuracy Comparison (Bar Chart)
        plt.subplot(2, 2, 3)
        accuracies = [self.move_accuracy(agent) * 100 for agent in agent_names]
        bars = plt.bar(agent_names, accuracies, color='#9C27B0')
        plt.ylabel('Accuracy (%)')
        plt.title('Move Accuracy')
        plt.ylim(0, 100)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        # Time per Move (Bar Chart)
        plt.subplot(2, 2, 4)
        times = [self.time_per_move(agent) for agent in agent_names]
        bars = plt.bar(agent_names, times, color='#FF9800')
        plt.ylabel('Seconds')
        plt.title('Average Time per Move')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}s',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        plt.tight_layout(pad=3.0)
        
        # Save or show the plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        
        if show:
            plt.show()
    
    def __str__(self):
        """Generate summary report of all agents"""
        report = []
        for agent, stats in self.agents.items():
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            total_moves = stats['legal_moves'] + stats.get('illegal_moves', 0)
            
            report.append(f"Agent: {agent}")
            report.append(f"  Games: {total_games} (W: {stats['wins']}, L: {stats['losses']}, D: {stats['draws']})")
            report.append(f"  Win Rate: {self.win_rate(agent):.2%}")
            report.append(f"  Move Accuracy: {self.move_accuracy(agent):.2%}")
            report.append(f"  Avg Time/Move: {self.time_per_move(agent):.4f}s")
            illegal = stats.get('illegal_moves', 0)
            report.append(f"  Total Moves: {total_moves} (Legal: {stats['legal_moves']}, Illegal: {illegal})")
        
        return "\n".join(report)