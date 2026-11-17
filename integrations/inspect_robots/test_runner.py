"""Offline wrapper checks using the actual upstream agent and a mock transport."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import inspect_robots
import inspect_robots_agent
from inspect_robots.guard import ClampGuard
from inspect_robots.mock import CubePickEmbodiment


SPEC = importlib.util.spec_from_file_location("inspect_mock_runner", Path(__file__).with_name("run_agent.py"))
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class AgentRunnerTests(unittest.TestCase):
    def run_main(self, arguments: list[str]):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_agent.py", *arguments]), contextlib.redirect_stdout(output):
            result = RUNNER.main()
        return result, output.getvalue()

    def test_opt_in_required_before_constructing_policy(self):
        with patch.object(inspect_robots_agent, "LLMAgentPolicy") as factory:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                self.run_main(["--model", "test/model", "--wire", "chat"])
            self.assertEqual(exit_info.exception.code, 2)
            factory.assert_not_called()

    def test_invalid_execution_budgets_and_seed_are_rejected(self):
        base = ["--model", "test/model", "--wire", "chat", "--allow-network"]
        for argument, value in (("--max-steps", "65"), ("--max-calls", "13"),
                                ("--max-calls", "0"), ("--seed", "-1"),
                                ("--seed", str(2**32))):
            with self.subTest(argument=argument, value=value):
                with patch.object(inspect_robots_agent, "LLMAgentPolicy") as factory:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                        self.run_main([*base, argument, value])
                    self.assertEqual(exit_info.exception.code, 2)
                    factory.assert_not_called()

    def test_custom_endpoint_and_credential_name_must_be_paired(self):
        for extra in (["--base-url", "https://llm.test/v1"], ["--api-key-env", "TEST_ONLY_KEY"]):
            with self.subTest(extra=extra), patch.object(inspect_robots_agent, "LLMAgentPolicy") as factory:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                    self.run_main(["--model", "test/model", "--wire", "chat", "--allow-network", *extra])
                self.assertEqual(exit_info.exception.code, 2)
                factory.assert_not_called()

    def test_real_agent_mock_transport_prompt_and_fixed_world(self):
        requests: list[dict] = []
        real_policy = inspect_robots_agent.LLMAgentPolicy

        def respond(request: httpx.Request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_done", "type": "function", "function": {
                    "name": "done", "arguments": json.dumps({"summary": "Stopped without reaching."})
                }}]
            }}]})

        def make_policy(**kwargs):
            return real_policy(**kwargs, transport=httpx.MockTransport(respond), env={})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root / "notes.txt"
            notes.write_text("TEST_VARIANT: inspect the target before moving.", encoding="utf-8")
            # A transport regression must fail locally instead of reaching any provider.
            with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
                 patch.object(inspect_robots_agent, "LLMAgentPolicy", side_effect=make_policy) as factory, \
                 patch.object(inspect_robots, "eval", wraps=inspect_robots.eval) as evaluate:
                code, output = self.run_main([
                    "--model", "test/model", "--wire", "chat", "--allow-network",
                    "--base-url", "https://llm.test/v1", "--api-key-env", "TEST_ONLY_KEY",
                    "--max-calls", "1", "--max-steps", "2", "--seed", "1",
                    "--notes-file", str(notes), "--eval-notes", "TEST_BRIEF: reach the cube.",
                    "--log-dir", str(root / "logs"),
                ])
            self.assertEqual(code, 0)
            summary = json.loads(output)
            self.assertEqual(summary["status"], "success")
            # A model claiming done is not sufficient to satisfy the environment scorer.
            self.assertEqual(summary["metrics"]["success_at_end"], 0.0)
            self.assertEqual(len(requests), 1)
            prompt = requests[0]["messages"][0]["content"]
            self.assertIn("TEST_VARIANT", prompt)
            self.assertIn("TEST_BRIEF", prompt)
            self.assertIn("Environment step budget", prompt)
            self.assertIn("data:image/png;base64,", json.dumps(requests[0]))
            task, _, embodiment = evaluate.call_args.args
            self.assertIsInstance(embodiment, CubePickEmbodiment)
            self.assertEqual(task.max_steps, 2)
            self.assertEqual(task.epochs, 1)
            self.assertIsInstance(evaluate.call_args.kwargs["guard"], ClampGuard)
            self.assertIsNone(evaluate.call_args.kwargs["grader"])
            self.assertIsNone(evaluate.call_args.kwargs["operator_input"])
            self.assertEqual(factory.call_args.kwargs["max_llm_calls"], 1)
            self.assertEqual(factory.call_args.kwargs["max_retries"], 1)
            self.assertEqual(factory.call_args.kwargs["timeout_s"], 30)


if __name__ == "__main__":
    unittest.main()
