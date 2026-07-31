"""CLI: python -m evals.run --trace-dir logs/traces --output report.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.metrics import Pricing
from evals.runner import BatchEvaluationRunner, write_report
from game.llm_client import create_client
from game.replay import ReplayRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=Path("logs/traces"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation/latest.json"),
    )
    parser.add_argument("--game-id", action="append", dest="game_ids")
    parser.add_argument("--input-price", type=float, default=0.0)
    parser.add_argument("--output-price", type=float, default=0.0)
    parser.add_argument(
        "--judge-model",
        help=(
            "Optional provider:model used as LLM Judge, for example "
            "anthropic:claude-sonnet-4-6"
        ),
    )
    parser.add_argument("--judge-max-actions", type=int, default=50)
    parser.add_argument("--experiment-label")
    parser.add_argument(
        "--metadata",
        default="{}",
        help="JSON object stored with the report for reproducibility",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "scenarios",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("--metadata must decode to a JSON object")
    judge_client = (
        create_client(args.judge_model) if args.judge_model else None
    )
    if judge_client is not None:
        judge_client.record_calls = False
    runner = BatchEvaluationRunner(
        ReplayRepository(args.trace_dir),
        pricing=Pricing(args.input_price, args.output_price),
        scenario_directory=args.scenario_dir,
        judge_client=judge_client,
        judge_max_actions=args.judge_max_actions,
    )
    report = runner.run(
        args.game_ids,
        experiment_label=args.experiment_label,
        metadata=metadata,
    )
    json_path, markdown_path = write_report(report, args.output)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
