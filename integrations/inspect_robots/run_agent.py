#!/usr/bin/env python3
"""Run the optional API agent on a fixed upstream in-memory CubePick world."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def bounded_int(low: int, high: int):
    def parse(value: str) -> int:
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number
    return parse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="exact provider/model identifier")
    parser.add_argument("--wire", required=True, choices=("chat", "responses", "messages"))
    parser.add_argument("--base-url", help="explicit compatible provider endpoint")
    parser.add_argument("--api-key-env", help="environment variable name, never the key itself")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--notes-file", type=Path, help="append advisory notes to upstream system prompt")
    parser.add_argument("--eval-notes", help="append a short evaluation brief to upstream system prompt")
    parser.add_argument("--max-steps", type=bounded_int(1, 64), default=32)
    parser.add_argument("--max-calls", type=bounded_int(1, 12), default=8)
    parser.add_argument("--seed", type=bounded_int(0, 2**32 - 1), default=0)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if not args.allow_network:
        parser.error("API calls require --allow-network; use smoke.py for an offline run")
    if args.base_url and not args.api_key_env:
        parser.error("--base-url requires --api-key-env to avoid an implicit credential choice")
    if args.api_key_env and not args.base_url:
        parser.error("--api-key-env requires --base-url; direct provider routing uses its documented key")

    from inspect_robots import Scene, Task, episode_length, eval, success_at_end
    from inspect_robots.guard import ClampGuard
    from inspect_robots.mock import CubePickEmbodiment
    from inspect_robots_agent import LLMAgentPolicy

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = args.log_dir or Path("runs/inspect_robots") / f"agent-{stamp}"
    task = Task(
        name="upstream-cubepick-api-agent",
        scenes=[Scene(id="reach", instruction="Reach the cube.", init_seed=args.seed)],
        scorer=[success_at_end(), episode_length()],
        max_steps=args.max_steps,
        epochs=1,
        metadata={"scope": "upstream_2d_reach_mock_not_robot_eval_point_sim"},
    )
    policy = LLMAgentPolicy(
        model=args.model,
        wire=args.wire,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_llm_calls=args.max_calls,
        timeout_s=30,
        # Upstream names this retries, but it is the total attempt count.
        max_retries=1,
        images="always",
        depth="off",
        transcript_echo=False,
        prior_learnings=str(args.notes_file.resolve()) if args.notes_file else None,
        eval_notes=args.eval_notes,
    )
    embodiment = CubePickEmbodiment(max_step=0.1)
    try:
        log, = eval(
            task,
            policy,
            embodiment,
            log_dir=str(log_dir),
            seed=args.seed,
            fail_on_error=True,
            guard=ClampGuard(embodiment.info.action_space),
            operator_input=None,
            grader=None,
            store_frames=True,
            environment_id="upstream:cubepick-2d-reach-mock",
            environment_revision="772019598e42ccc7921a043dee3a2ea2692020ea",
        )
    finally:
        embodiment.close()
    print(json.dumps({
        "status": log.status,
        "scope": "upstream_2d_reach_mock",
        "metrics": log.results.metrics,
        "error": log.error,
        "log_dir": str(log_dir.resolve()),
    }, indent=2))
    # Completion of the eval and achievement of its task are separate outcomes.
    return 0 if log.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
