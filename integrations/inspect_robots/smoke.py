#!/usr/bin/env python3
"""Bounded, offline upstream CubePick smoke; no registry/config selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()

    from inspect_robots import Scene, Task, episode_length, eval, success_at_end
    from inspect_robots.guard import ClampGuard
    from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = args.log_dir or Path("runs/inspect_robots") / f"smoke-{stamp}"
    task = Task(
        name="upstream-cubepick-smoke",
        scenes=[
            Scene(id="seed-0", instruction="Reach the cube.", init_seed=0),
            Scene(id="seed-1", instruction="Reach the cube.", init_seed=1),
        ],
        scorer=[success_at_end(), episode_length()],
        max_steps=32,
        epochs=2,
        metadata={"scope": "upstream_2d_reach_mock_not_robot_eval_point_sim"},
    )
    embodiment = CubePickEmbodiment(max_step=0.1)
    try:
        log, = eval(
            task,
            ScriptedPolicy(chunk_size=4, max_step=0.1),
            embodiment,
            log_dir=str(log_dir),
            seed=0,
            fail_on_error=True,
            guard=ClampGuard(embodiment.info.action_space),
            operator_input=None,
            grader=None,
            store_frames=False,
            environment_id="upstream:cubepick-2d-reach-mock",
            environment_revision="772019598e42ccc7921a043dee3a2ea2692020ea",
        )
    finally:
        embodiment.close()

    passed = (
        log.status == "success"
        and log.results.total_trials == 4
        and log.results.errored_trials == 0
        and log.results.metrics.get("success_at_end") == 1.0
    )
    print(json.dumps({
        "smoke_passed": passed,
        "scope": "upstream_2d_reach_mock",
        "trials": log.results.total_trials,
        "metrics": log.results.metrics,
        "log_dir": str(log_dir.resolve()),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
