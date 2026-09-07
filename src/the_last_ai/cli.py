"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from the_last_ai.experiments.architecture_compare import run_architecture_comparison
from the_last_ai.experiments.communication_experiment import run_experiment_8
from the_last_ai.experiments.computational_loss import run_experiment_9
from the_last_ai.experiments.counterfactual_reasoning import run_counterfactual_reasoning
from the_last_ai.experiments.entity_representation import run_entity_representation
from the_last_ai.experiments.gradual_population import (
    run_gradual_population_reduction,
    run_the_last_ai,
)
from the_last_ai.experiments.learned_environment import run_learned_environment
from the_last_ai.experiments.random_baseline import run_random_baseline
from the_last_ai.experiments.restoration import run_restoration
from the_last_ai.experiments.social_memory_propagation import run_social_memory_propagation
from the_last_ai.experiments.sudden_disappearance import run_sudden_disappearance
from the_last_ai.experiments.world_model_prediction import run_world_model_experiment
from the_last_ai.simulation.engine import Simulation, SimulationConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="last-ai", description="The Last AI simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a short simulation")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--ticks", type=int, default=50)
    run_parser.add_argument("--agents", type=int, default=5)
    run_parser.add_argument("--width", type=int, default=20)
    run_parser.add_argument("--height", type=int, default=12)
    run_parser.add_argument(
        "--agent-type",
        choices=[
            "RandomAgent",
            "ValueBasedAgent",
            "MemoryAgent",
            "SocialAgent",
            "PredictiveAgent",
        ],
        default="RandomAgent",
    )
    run_parser.add_argument("--resources", type=int, default=0)
    run_parser.add_argument("--save", type=Path, default=None)

    def _add_seed_ticks(p, ticks=200, agents=8):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--ticks", type=int, default=ticks)
        p.add_argument("--agents", type=int, default=agents)
        p.add_argument("--output-dir", type=Path)

    exp0 = sub.add_parser("experiment-0", help="Random baseline")
    _add_seed_ticks(exp0)
    exp0.set_defaults(output_dir=Path("outputs/experiment_0"))

    exp1 = sub.add_parser("experiment-1", help="Learned environment")
    exp1.add_argument("--seed", type=int, default=0)
    exp1.add_argument("--ticks", type=int, default=300)
    exp1.add_argument("--agents", type=int, default=4)
    exp1.add_argument("--resources", type=int, default=10)
    exp1.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_1"))

    exp2 = sub.add_parser("experiment-2", help="Entity representation")
    exp2.add_argument("--seed", type=int, default=0)
    exp2.add_argument("--familiar-ticks", type=int, default=120)
    exp2.add_argument("--unfamiliar-ticks", type=int, default=8)
    exp2.add_argument("--post-ticks", type=int, default=80)
    exp2.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_2"))

    exp3 = sub.add_parser("experiment-3", help="Sudden social disappearance")
    exp3.add_argument("--seed", type=int, default=0)
    exp3.add_argument("--familiar-ticks", type=int, default=100)
    exp3.add_argument("--unfamiliar-ticks", type=int, default=6)
    exp3.add_argument("--post-ticks", type=int, default=60)
    exp3.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_3"))

    exp4 = sub.add_parser("experiment-4", help="Social memory propagation")
    exp4.add_argument("--seed", type=int, default=0)
    exp4.add_argument("--strong-ticks", type=int, default=40)
    exp4.add_argument("--weak-ticks", type=int, default=1)
    exp4.add_argument("--test-ticks", type=int, default=50)
    exp4.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_4"))

    exp_wm = sub.add_parser("experiment-wm", help="World-model prediction")
    exp_wm.add_argument("--seed", type=int, default=0)
    exp_wm.add_argument("--ticks", type=int, default=150)
    exp_wm.add_argument("--agents", type=int, default=5)
    exp_wm.add_argument("--resources", type=int, default=8)
    exp_wm.add_argument("--disappear-at", type=int, default=90)
    exp_wm.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_world_model"))

    exp5 = sub.add_parser("experiment-5", help="Gradual population reduction (small)")
    exp5.add_argument("--seed", type=int, default=0)
    exp5.add_argument("--agents", type=int, default=20)
    exp5.add_argument("--ticks-between", type=int, default=40)
    exp5.add_argument("--schedule", type=str, default="10,5,2,1")
    exp5.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_5"))

    exp6 = sub.add_parser("experiment-6", help="Restoration after disappearance")
    exp6.add_argument("--seed", type=int, default=0)
    exp6.add_argument("--formation-ticks", type=int, default=40)
    exp6.add_argument("--absence-ticks", type=int, default=30)
    exp6.add_argument("--restoration-ticks", type=int, default=40)
    exp6.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_6"))

    exp7 = sub.add_parser("experiment-7", help="Counterfactual multi-step reasoning")
    exp7.add_argument("--seed", type=int, default=0)
    exp7.add_argument("--ticks", type=int, default=80)
    exp7.add_argument("--agents", type=int, default=5)
    exp7.add_argument("--resources", type=int, default=8)
    exp7.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_7"))

    exp8 = sub.add_parser(
        "experiment-8",
        help="Primitive symbolic communication (send/receive/verify/collapse)",
    )
    exp8.add_argument("--seed", type=int, default=None)
    exp8.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (overrides --seed when set)",
    )
    exp8.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_8"))

    exp9 = sub.add_parser(
        "experiment-9",
        help="Computational loss after social disappearance (relationship-dependent)",
    )
    exp9.add_argument("--seed", type=int, default=None)
    exp9.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (overrides --seed when set)",
    )
    exp9.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_9"))

    last = sub.add_parser(
        "the-last-ai",
        help="Centrepiece collapse: 100→50→25→10→5→2→1 with mind snapshots",
    )
    last.add_argument("--seed", type=int, default=0)
    last.add_argument("--agents", type=int, default=100)
    last.add_argument("--schedule", type=str, default="50,25,10,5,2,1")
    last.add_argument("--ticks-between", type=int, default=30)
    last.add_argument("--final-ticks", type=int, default=40)
    last.add_argument("--resources", type=int, default=40)
    last.add_argument("--width", type=int, default=40)
    last.add_argument("--height", type=int, default=30)
    last.add_argument("--output-dir", type=Path, default=Path("outputs/the_last_ai"))

    interpret = sub.add_parser(
        "interpret",
        help="Turn experiment/mind JSON into a human-readable narrative",
    )
    interpret.add_argument(
        "path",
        type=Path,
        help="Path to report_seed*.json, restoration JSON, or a mind snapshot",
    )
    interpret.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional markdown output path (default: same name with .md)",
    )
    interpret.add_argument(
        "--no-minds",
        action="store_true",
        help="For Last AI reports, skip loading the final mind snapshot",
    )

    inspect_p = sub.add_parser(
        "inspect-agent",
        help="Export Individual Agent Life Observatory biography for one agent",
    )
    inspect_p.add_argument(
        "path",
        type=Path,
        help="final_sim_seed*.json, agent_*_life.json, or experiment output directory",
    )
    inspect_p.add_argument("agent_id", type=str, help="e.g. agent_000")
    inspect_p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for life.json / life.txt / life.html",
    )

    compare = sub.add_parser(
        "compare-architectures",
        help="Multi-seed frozen-architecture comparison across agent types",
    )
    compare.add_argument("--seeds", type=str, default="0,1,2,3,4")
    compare.add_argument("--agents", type=int, default=24)
    compare.add_argument("--schedule", type=str, default="12,6,3,1")
    compare.add_argument("--ticks-between", type=int, default=15)
    compare.add_argument("--final-ticks", type=int, default=20)
    compare.add_argument(
        "--architectures",
        type=str,
        default="RandomAgent,ValueBasedAgent,MemoryAgent,SocialAgent,PredictiveAgent",
    )
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/architecture_compare"))

    args = parser.parse_args(argv)

    if args.command == "run":
        config = SimulationConfig(
            seed=args.seed,
            n_agents=args.agents,
            width=args.width,
            height=args.height,
            agent_type=args.agent_type,
            n_resources=args.resources,
        )
        sim = Simulation.create(config)
        metrics = sim.run(args.ticks)
        print(sim.render())
        print()
        print(metrics.summary())
        if args.agent_type != "RandomAgent":
            print(sim.agent_learning_summary())
        if args.save:
            print(f"Saved snapshot to {sim.save(args.save)}")
        return 0

    if args.command == "experiment-0":
        result = run_random_baseline(
            seed=args.seed, n_ticks=args.ticks, n_agents=args.agents, output_dir=args.output_dir
        )
        print(result["summary"])
        return 0

    if args.command == "experiment-1":
        result = run_learned_environment(
            seed=args.seed,
            n_ticks=args.ticks,
            n_agents=args.agents,
            n_resources=args.resources,
            output_dir=args.output_dir,
        )
        print(result["comparison"]["delta"])
        return 0

    if args.command == "experiment-2":
        result = run_entity_representation(
            seed=args.seed,
            familiar_ticks=args.familiar_ticks,
            unfamiliar_ticks=args.unfamiliar_ticks,
            post_ticks=args.post_ticks,
            output_dir=args.output_dir,
        )
        print(result["comparison"]["deltas"])
        return 0

    if args.command == "experiment-3":
        result = run_sudden_disappearance(
            seed=args.seed,
            familiar_ticks=args.familiar_ticks,
            unfamiliar_ticks=args.unfamiliar_ticks,
            post_ticks=args.post_ticks,
            output_dir=args.output_dir,
        )
        print(result["comparison"]["deltas"])
        return 0

    if args.command == "experiment-4":
        result = run_social_memory_propagation(
            seed=args.seed,
            strong_ticks=args.strong_ticks,
            weak_ticks=args.weak_ticks,
            test_ticks=args.test_ticks,
            output_dir=args.output_dir,
        )
        print(result["summary"])
        return 0

    if args.command == "experiment-wm":
        result = run_world_model_experiment(
            seed=args.seed,
            n_ticks=args.ticks,
            n_agents=args.agents,
            n_resources=args.resources,
            disappear_at=args.disappear_at,
            output_dir=args.output_dir,
        )
        print(result["comparison"]["changing_world"]["final_mean_error_ema"])
        return 0

    if args.command == "experiment-5":
        schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
        result = run_gradual_population_reduction(
            seed=args.seed,
            initial_agents=args.agents,
            schedule=schedule,
            ticks_between=args.ticks_between,
            output_dir=args.output_dir,
        )
        print(result["snapshot"]["final_population"])
        return 0

    if args.command == "experiment-6":
        result = run_restoration(
            seed=args.seed,
            formation_ticks=args.formation_ticks,
            absence_ticks=args.absence_ticks,
            restoration_ticks=args.restoration_ticks,
            output_dir=args.output_dir,
        )
        r = result["result"]
        print(
            {
                "residual_expectation_rate": r["during_absence"]["residual_expectation_rate"],
                "adaptation_ticks": r["after_restoration"]["adaptation_ticks_to_90pct_trust"],
                "memory_persisted": r["during_absence"]["memory_persisted"],
            }
        )
        return 0

    if args.command == "experiment-7":
        result = run_counterfactual_reasoning(
            seed=args.seed,
            n_ticks=args.ticks,
            n_agents=args.agents,
            n_resources=args.resources,
            output_dir=args.output_dir,
        )
        print({"best_plan": result["result"]["plan_horizon_2"].get("best")})
        return 0

    if args.command == "experiment-8":
        seeds = None
        if args.seeds:
            seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        result = run_experiment_8(
            seed=args.seed if seeds is None else None,
            seeds=seeds if seeds is not None else ([args.seed] if args.seed is not None else [0]),
            output_dir=args.output_dir,
        )
        print(result["report"]["aggregates_enabled"])
        print(f"Wrote {result['report_path']}")
        return 0

    if args.command == "experiment-9":
        seeds = None
        if args.seeds:
            seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        result = run_experiment_9(
            seed=args.seed if seeds is None else None,
            seeds=seeds if seeds is not None else ([args.seed] if args.seed is not None else [0]),
            output_dir=args.output_dir,
        )
        print(result["report"]["aggregates"])
        print(f"Wrote {result['report_path']}")
        return 0

    if args.command == "the-last-ai":
        schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
        result = run_the_last_ai(
            seed=args.seed,
            initial_agents=args.agents,
            schedule=schedule,
            ticks_between=args.ticks_between,
            final_ticks=args.final_ticks,
            n_resources=args.resources,
            width=args.width,
            height=args.height,
            output_dir=args.output_dir,
        )
        print(result["report"]["control_comparison"])
        print(f"Wrote {result['report_path']}")
        if result.get("interpretation_path"):
            print(f"Wrote {result['interpretation_path']}")
        if result.get("observatory_paths"):
            print(f"Observatory: {result['observatory_paths']}")
        return 0

    if args.command == "interpret":
        from the_last_ai.experiments.interpret import interpret_json_file, write_interpretation

        text = interpret_json_file(args.path, load_event_minds=not args.no_minds)
        out = write_interpretation(
            args.path,
            output_path=args.out,
            load_event_minds=not args.no_minds,
        )
        print(text)
        print(f"Wrote {out}")
        return 0

    if args.command == "inspect-agent":
        from the_last_ai.observatory.inspect import inspect_agent_from_path

        result = inspect_agent_from_path(
            args.path, args.agent_id, output_dir=args.out_dir
        )
        print(result.get("overview"))
        print(result.get("paths"))
        return 0

    if args.command == "compare-architectures":
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
        architectures = [x.strip() for x in args.architectures.split(",") if x.strip()]
        result = run_architecture_comparison(
            seeds=seeds,
            initial_agents=args.agents,
            schedule=schedule,
            ticks_between=args.ticks_between,
            final_ticks=args.final_ticks,
            architectures=architectures,
            output_dir=args.output_dir,
        )
        print(result["aggregates"])
        print(f"Wrote {result['path']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
