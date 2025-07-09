import pytest
import os
from metrics.metrics import GameMetrics

# -- Agent Initialization Tests -- #


def test_add_agent_in_agents_dict():
    """Test that a new agent is added to the agents dictionary."""
    gm = GameMetrics()
    gm.add_agent("AgentA")
    assert "AgentA" in gm.agents


def test_add_agent_initial_metrics():
    """Test that a new agent's metrics are initialized correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentA")
    stats = gm.agents["AgentA"]
    assert stats == {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_time": 0.0,
        "legal_moves": 0,
        "illegal_moves": 0,
    }


# -- Move Recording and Accuracy Tests -- #


def test_record_move_updates_total_time():
    """Test that total_time is correctly updated when moves are recorded."""
    gm = GameMetrics()
    gm.add_agent("AgentB")
    gm.record_move("AgentB", 1.5, is_legal=True)
    gm.record_move("AgentB", 2.0, is_legal=False)
    assert gm.agents["AgentB"]["total_time"] == 3.5


def test_record_move_counts_legal_moves():
    """Test that legal moves are counted correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentB")
    gm.record_move("AgentB", 1.5, is_legal=True)
    gm.record_move("AgentB", 2.0, is_legal=False)
    assert gm.agents["AgentB"]["legal_moves"] == 1


def test_record_move_counts_illegal_moves():
    """Test that illegal moves are counted correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentB")
    gm.record_move("AgentB", 1.5, is_legal=True)
    gm.record_move("AgentB", 2.0, is_legal=False)
    assert gm.agents["AgentB"]["illegal_moves"] == 1


def test_move_accuracy_calculation():
    """Test that move accuracy is calculated correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentB")
    gm.record_move("AgentB", 1.5, is_legal=True)
    gm.record_move("AgentB", 2.0, is_legal=False)
    assert gm.move_accuracy("AgentB") == 0.5


# -- Result Recording and Win Rate Tests --


def test_record_result_counts_wins():
    """Test that wins are counted correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentC")
    gm.record_result("AgentC", "win")
    gm.record_result("AgentC", "win")
    assert gm.agents["AgentC"]["wins"] == 2


def test_record_result_counts_losses():
    """Test that losses are counted correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentC")
    gm.record_result("AgentC", "loss")
    assert gm.agents["AgentC"]["losses"] == 1


def test_record_result_counts_draws():
    """Test that draws are counted correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentC")
    gm.record_result("AgentC", "draw")
    assert gm.agents["AgentC"]["draws"] == 1


def test_win_rate_calculation():
    """Test that win rate is calculated correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentC")
    gm.record_result("AgentC", "win")
    gm.record_result("AgentC", "loss")
    gm.record_result("AgentC", "draw")
    gm.record_result("AgentC", "win")
    assert gm.win_rate("AgentC") == 2 / 4


# -- Time Per Move Test -- #


def test_time_per_move_calculation():
    """Test that average time per move is calculated correctly."""
    gm = GameMetrics()
    gm.add_agent("AgentD")
    gm.record_move("AgentD", 2.0, is_legal=True)
    gm.record_move("AgentD", 4.0, is_legal=True)
    gm.record_move("AgentD", 4.0, is_legal=False)
    assert gm.time_per_move("AgentD") == 10.0 / 3


# -- Unknown Agent Metrics -- #


def test_win_rate_unknown_agent():
    """Test that win rate for unknown agent is 0.0."""
    gm = GameMetrics()
    assert gm.win_rate("Unknown") == 0.0


def test_move_accuracy_unknown_agent():
    """Test that move accuracy for unknown agent is 0.0."""
    gm = GameMetrics()
    assert gm.move_accuracy("Unknown") == 0.0


def test_time_per_move_unknown_agent():
    """Test that time per move for unknown agent is 0.0."""
    gm = GameMetrics()
    assert gm.time_per_move("Unknown") == 0.0


# -- String Summary Tests -- #


def test_summary_includes_agent_name():
    """Test that agent name appears in summary string."""
    gm = GameMetrics()
    gm.add_agent("AgentE")
    gm.record_result("AgentE", "win")
    gm.record_move("AgentE", 1.0, is_legal=True)
    gm.record_move("AgentE", 2.0, is_legal=False)
    summary = str(gm)
    assert "Agent: AgentE" in summary


def test_summary_includes_game_results():
    """Test that summary string includes game results."""
    gm = GameMetrics()
    gm.add_agent("AgentE")
    gm.record_result("AgentE", "win")
    summary = str(gm)
    assert "Games: 1 (W: 1, L: 0, D: 0)" in summary


def test_summary_includes_win_rate():
    """Test that win rate is included in summary string."""
    gm = GameMetrics()
    gm.add_agent("AgentE")
    gm.record_result("AgentE", "win")
    summary = str(gm)
    assert "Win Rate: 100.00%" in summary


def test_summary_includes_move_accuracy():
    """Test that move accuracy is included in summary string."""
    gm = GameMetrics()
    gm.add_agent("AgentE")
    gm.record_move("AgentE", 1.0, is_legal=True)
    gm.record_move("AgentE", 2.0, is_legal=False)
    summary = str(gm)
    assert "Move Accuracy: 50.00%" in summary


def test_summary_includes_avg_time_per_move():
    """Test that average time per move is included in summary string."""
    gm = GameMetrics()
    gm.add_agent("AgentE")
    gm.record_move("AgentE", 1.0, is_legal=True)
    gm.record_move("AgentE", 2.0, is_legal=False)
    summary = str(gm)
    assert "Avg Time/Move: 1.5000s" in summary


def test_summary_includes_move_counts():
    """Test that total legal and illegal move counts are in summary string."""
    gm = GameMetrics()
    gm.add_agent("X")
    gm.record_move("X", 1.0, is_legal=True)
    gm.record_move("X", 2.0, is_legal=False)
    summary = str(gm)
    assert "Total Moves: 2 (Legal: 1, Illegal: 1)" in summary


# -- Plotting Tests -- #


def test_plot_results_creates_file(tmp_path):
    """Test that plot_results creates a plot file."""
    gm = GameMetrics()
    gm.add_agent("X")
    gm.record_result("X", "win")
    gm.record_move("X", 1.0, is_legal=True)
    save_path = tmp_path / "metrics_plot.png"
    gm.plot_results(save_path=str(save_path), show=False)
    assert os.path.exists(save_path)


def test_plot_results_no_agents_prints_message(capsys):
    """Test that a message is printed if plot_results is called with no agents."""
    gm = GameMetrics()
    gm.plot_results(show=False)
    captured = capsys.readouterr()
    assert "No metrics to plot" in captured.out


def test_plot_performance_radar_runs_without_error():
    """Test that plot_performance_radar runs without raising exceptions."""
    gm = GameMetrics()
    gm.add_agent("X")
    gm.record_result("X", "win")
    gm.record_move("X", 1.0, is_legal=True)
    gm.add_agent("Y")
    gm.record_result("Y", "loss")
    gm.record_move("Y", 2.0, is_legal=False)
    # Should not raise any exceptions even if plt.show() opens a window
    try:
        gm.plot_performance_radar()
    except Exception as e:
        pytest.fail(f"plot_performance_radar raised an exception: {e}")


def test_plot_performance_radar_single_agent():
    """Test radar plot with only one agent."""
    gm = GameMetrics()
    gm.add_agent("SoloAgent")
    gm.record_result("SoloAgent", "win")
    gm.record_move("SoloAgent", 1.5, is_legal=True)
    try:
        gm.plot_performance_radar()
    except Exception as e:
        pytest.fail(f"plot_performance_radar with single agent raised: {e}")


def test_plot_move_duration_distribution_runs_without_error():
    """Test that plot_move_duration_distribution runs without raising exceptions."""
    gm = GameMetrics()
    gm.add_agent("AgentPlot")
    gm.record_move("AgentPlot", 1.0, is_legal=True)
    gm.record_move("AgentPlot", 2.0, is_legal=True)
    try:
        gm.plot_move_duration_distribution("AgentPlot")
    except Exception as e:
        pytest.fail(f"plot_move_duration_distribution raised an exception: {e}")


def test_plot_move_duration_distribution_no_moves(capsys):
    """Test that a message is printed if agent has no move times."""
    gm = GameMetrics()
    gm.add_agent("Empty")
    gm.plot_move_duration_distribution("Empty")
    captured = capsys.readouterr()
    assert "No move times recorded for agent 'EmptyAgent'" in captured.out
