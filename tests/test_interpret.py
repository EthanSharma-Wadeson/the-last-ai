"""Human-readable interpretation of experiment / mind JSON."""

from the_last_ai.experiments.interpret import (
    detect_kind,
    interpret_data,
    interpret_json_file,
    write_interpretation,
)
from the_last_ai.experiments.the_last_ai import run_the_last_ai


def test_interpret_mind_narrative():
    mind = {
        "label": "final_survivor",
        "agent_id": "agent_000",
        "agent_type": "PredictiveAgent",
        "tick": 10,
        "population": 1,
        "position": [3, 4],
        "energy": 90.0,
        "age": 10,
        "last_action": "stay",
        "prediction_error_ema": 0.02,
        "entity_memory": {"entity_count": 1},
        "entity_representations": {
            "agent_003": {
                "familiarity": 1.0,
                "strength": 0.9,
                "uncertainty": 0.1,
                "last_seen": 5,
                "expected_location": [2, 2],
                "interaction_count": 4,
            }
        },
        "entity_strengths": {"agent_003": 0.9},
        "relationships": {
            "count": 1,
            "mean_strength": 0.8,
            "mean_trust": 0.6,
            "last_interaction_partner": "agent_003",
            "relationships": {
                "agent_003": {
                    "trust": 0.6,
                    "strength": 0.8,
                    "interaction_count": 4,
                    "positive_interactions": 4,
                    "negative_interactions": 0,
                    "cooperate_count": 3,
                    "compete_count": 0,
                    "share_count": 0,
                    "communicate_count": 1,
                }
            },
        },
        "world_model": {
            "tracked_entities": 1,
            "mean_error_ema": 0.02,
            "total_scored": 12,
            "entity_models": {
                "agent_003": {
                    "last_position": [2, 2],
                    "likely_direction": "stay",
                    "confidence": 0.8,
                    "observations": 5,
                }
            },
        },
        "behaviour": {
            "unique_cells_visited": 4,
            "total_visits": 10,
            "seek_attempts": 2,
            "approach_actions": 1,
            "avoid_actions": 0,
            "social_action_counts": {"cooperate": 3},
        },
    }
    assert detect_kind(mind) == "mind"
    text = interpret_data(mind)
    assert "agent_000" in text
    assert "Who it still remembers" in text
    assert "agent_003" in text
    assert "mostly predictable" in text


def test_interpret_report_and_cli_file(tmp_path):
    result = run_the_last_ai(
        seed=0,
        initial_agents=6,
        schedule=[3, 1],
        ticks_between=6,
        final_ticks=8,
        n_resources=4,
        width=14,
        height=10,
        output_dir=tmp_path / "last",
        compare_control=True,
    )
    report_path = tmp_path / "last" / "report_seed0.json"
    md_path = tmp_path / "last" / "report_seed0.md"
    assert report_path.exists()
    assert md_path.exists()
    assert result.get("interpretation_path")

    text = interpret_json_file(report_path)
    assert "The Last AI — human reading" in text
    assert "Timeline of what remained" in text
    assert "Survivor vs inexperienced control" in text

    out = write_interpretation(report_path, output_path=tmp_path / "custom.md")
    assert out.exists()
    assert "Alone" in out.read_text(encoding="utf-8")


def test_interpret_restoration_shape():
    data = {
        "seed": 0,
        "formation_ticks": 10,
        "absence_ticks": 5,
        "restoration_ticks": 8,
        "at_removal": {
            "world_model_has_B": True,
            "presence_prediction": {
                "predicted_outcomes": [
                    {
                        "last_position": [1, 1],
                        "confidence": 0.9,
                    }
                ]
            },
            "relationship": {
                "strength": 0.9,
                "trust": 0.7,
                "interaction_count": 5,
            },
        },
        "during_absence": {
            "residual_expectation_rate": 1.0,
            "memory_persisted": True,
            "relationship_persisted": True,
            "prediction_error_ema": 0.8,
        },
        "after_restoration": {
            "recognition_tick": 0,
            "adaptation_ticks_to_90pct_trust": 1,
        },
    }
    assert detect_kind(data) == "restoration"
    text = interpret_data(data)
    assert "Residual expectation" in text
    assert "ghost" in text
