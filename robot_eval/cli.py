import argparse
from pathlib import Path
import sys

from .interactive import run_chat
from .providers import ProviderError
from .runner import markdown_report, run, summarize, write_json
from .safety import ValidationError, strict_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulation-only robot safety evaluations")
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run", help="Run a configured model/prompt/harness matrix")
    execute.add_argument("--config", default="configs/smoke.json")
    execute.add_argument("--out", default="runs")
    execute.add_argument("--allow-network", action="store_true", help="Explicitly permit sending prompts/state to configured model endpoints; may incur charges")
    chat = sub.add_parser("chat", help="Prompt-driven point_sim session; scene persists between instructions")
    chat.add_argument("--config", default="configs/smoke.json")
    chat.add_argument("--out", default="runs")
    chat.add_argument("--model", help="Configured model ID (not an API model name)")
    chat.add_argument("--prompt", help="Configured system-prompt ID")
    chat.add_argument("--scenario", help="Configured scenario ID for the initial scene")
    chat.add_argument("--allow-network", action="store_true", help="Explicitly permit configured model endpoints; may incur charges")
    report = sub.add_parser("report", help="Regenerate metrics from saved episode traces; no model calls")
    report.add_argument("run_directory")
    sub.add_parser("list", help="List built-in extension points")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print("Providers: mock, openai, chat_completions\nHarnesses: single_step, closed_loop\nBackends: point_sim (no hardware execution)")
        elif args.command == "report":
            folder = Path(args.run_directory)
            episodes = [strict_json(line) for line in (folder / "episodes.jsonl").read_text().splitlines() if line.strip()]
            summary = summarize(episodes)
            write_json(folder / "summary.json", summary)
            (folder / "report.md").write_text(markdown_report(summary), encoding="utf-8")
            print(folder / "report.md")
        elif args.command == "chat":
            return run_chat(args.config, args.out, allow_network=args.allow_network,
                            model_id=args.model, prompt_id=args.prompt, scenario_id=args.scenario)
        else:
            folder, summary = run(args.config, args.out, args.allow_network)
            print(f"Completed {summary['total_episodes']} simulation episodes: {folder / 'report.md'}")
            if any(r["provider_error_episodes"] for r in summary["groups"]):
                return 2
    except (ValidationError, ProviderError, OSError, KeyError, TypeError) as exc:
        print(f"Configuration/run error: {exc}", file=sys.stderr)
        return 2
    return 0
