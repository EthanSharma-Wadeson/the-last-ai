"""Experiment 0 smoke test."""

from pathlib import Path

from the_last_ai.experiments.random_baseline import run_random_baseline


def test_random_baseline_writes_outputs(tmp_path: Path):
    result = run_random_baseline(
        seed=3,
        n_ticks=40,
        n_agents=5,
        output_dir=tmp_path / "exp0",
    )
    assert result["summary"]["ticks_recorded"] == 40
    assert result["summary"]["final_population"] == 5
    assert Path(result["snapshot_path"]).exists()
    assert Path(result["summary_path"]).exists()
