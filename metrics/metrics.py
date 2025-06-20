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
            
        total_moves = agent['legal_moves'] + agent['illegal_moves']
        return agent['total_time'] / total_moves if total_moves else 0.0
    
    def move_accuracy(self, agent_name: str) -> float:
        """Calculate move accuracy for specified agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return 0.0
            
        total_moves = agent['legal_moves'] + agent['illegal_moves']
        return agent['legal_moves'] / total_moves if total_moves else 0.0
    
    def __str__(self):
        """Generate summary report of all agents"""
        report = []
        for agent, stats in self.agents.items():
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            total_moves = stats['legal_moves'] + stats['illegal_moves']
            
            report.append(f"Agent: {agent}")
            report.append(f"  Games: {total_games} (W: {stats['wins']}, L: {stats['losses']}, D: {stats['draws']})")
            report.append(f"  Win Rate: {self.win_rate(agent):.2%}")
            report.append(f"  Move Accuracy: {self.move_accuracy(agent):.2%}")
            report.append(f"  Avg Time/Move: {self.time_per_move(agent):.4f}s")
            report.append(f"  Total Moves: {total_moves} (Legal: {stats['legal_moves']}, Illegal: {stats['illegal_moves']})")
        
        return "\n".join(report)