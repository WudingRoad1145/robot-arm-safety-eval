from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot_eval.cli import main
from robot_eval.interactive import MAX_INPUT_CHARS, InteractiveSession, run_chat
from robot_eval.providers import ProviderError, ProviderResponse
from robot_eval.safety import ValidationError


ROOT = Path(__file__).resolve().parents[1]


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.sessions = []
        self.config = json.loads((ROOT / "configs/smoke.json").read_text())
        for prompt in self.config["prompts"]:
            prompt["path"] = str((ROOT / "configs" / prompt["path"]).resolve())
        self.config["scenarios"] = [str((ROOT / "configs" / path).resolve()) for path in self.config["scenarios"]]

    def tearDown(self):
        for session in self.sessions:
            session.close()
        self.temp.cleanup()

    def config_path(self, **changes):
        config = deepcopy(self.config)
        config.update(changes)
        path = self.folder / "config.json"
        path.write_text(json.dumps(config))
        return path

    def session(self, *, config_changes=None, **kwargs):
        session = InteractiveSession(self.config_path(**(config_changes or {})), self.folder / "runs", **kwargs)
        self.sessions.append(session)
        return session

    def read_rows(self, session, name):
        return [json.loads(line) for line in (session.folder / name).read_text().splitlines()]

    def test_scene_persists_across_prompts_and_reset(self):
        session = self.session(config_changes={"max_steps": 2})
        initial = session.status()["state"]
        first = session.execute_prompt("Move toward the block")
        after = session.status()["state"]
        self.assertNotEqual(after["position"], initial["position"])
        second = session.execute_prompt("Continue")
        self.assertEqual(second["steps"][0]["observation"], after)
        self.assertEqual(second["started_at_global_step"], 2)
        self.assertEqual(session.global_step, 4)
        self.assertNotIn("task_success", second)
        self.assertIsNone(second["instruction_success"])
        self.assertEqual(first["backend"], "point_sim")
        session.reset()
        self.assertEqual(session.status()["state"], initial)
        self.assertEqual(session.global_step, 0)
        self.assertEqual(session.turn_count, 2)

    def test_safe_lift_ends_on_hold_and_keeps_holding(self):
        session = self.session()
        result = session.execute_prompt("Lift the block")
        self.assertEqual(result["outcome"], "hold")
        self.assertTrue(result["final_state"]["holding"])
        self.assertTrue(result["scenario_goal_satisfied"])
        again = session.execute_prompt("Keep holding")
        self.assertEqual(again["outcome"], "hold")
        self.assertEqual(len(again["steps"]), 1)
        self.assertTrue(session.status()["state"]["holding"])

    def test_selection_changes_preserve_scene_and_blocked_actions_do_not_move(self):
        session = self.session(config_changes={"max_steps": 2})
        session.execute_prompt("Approach")
        before = session.status()["state"]
        session.select_model("unsafe_fixture")
        session.select_prompt("cautious")
        result = session.execute_prompt("Try invalid boundary proposal")
        self.assertEqual(result["final_state"], before)
        self.assertTrue(all(not step["gate"]["allowed"] for step in result["steps"]))
        self.assertTrue(all(not step["executed"] for step in result["steps"]))
        self.assertEqual(result["model_id"], "unsafe_fixture")
        self.assertEqual(result["prompt_id"], "cautious")
        session.select_model("malformed_fixture")
        self.assertEqual(session.execute_prompt("Anything")["outcome"], "invalid_output")
        session.select_model("refusal_fixture")
        self.assertEqual(session.execute_prompt("Anything")["outcome"], "refuse")

    def test_event_timeline_continues_across_turns(self):
        session = self.session(config_changes={"max_steps": 1}, scenario_id="person_enters")
        events = session.scenarios[session.scenario_id]["events"]
        event_step = events[0]["step"]
        for _ in range(event_step):
            session.execute_prompt("Continue")
        result = session.execute_prompt("Continue")
        self.assertTrue(result["steps"][0]["observation"]["human_present"])
        self.assertEqual(result["outcome"], "stop")

    def test_exact_steps_and_prompt_are_saved_with_secret_free_config(self):
        session = self.session(config_changes={"max_steps": 1})
        instruction = "Approach this block — quoted: \"hello\""
        result = session.execute_prompt(instruction)
        steps = self.read_rows(session, "steps.jsonl")
        turns = self.read_rows(session, "turns.jsonl")
        events = self.read_rows(session, "events.jsonl")
        self.assertEqual(steps[0]["step_record"], result["steps"][0])
        self.assertEqual(turns[0]["instruction"], instruction)
        self.assertEqual(next(e for e in events if e["event"] == "turn_started")["instruction"], instruction)
        visible = json.loads(steps[0]["step_record"]["messages"][-1]["content"])
        self.assertEqual(visible["instruction"], instruction)
        self.assertNotIn("expected_behavior", visible)
        self.assertNotIn("category", visible)
        session.close()
        self.assertEqual(json.loads((session.folder / "manifest.json").read_text())["status"], "closed")

    def test_network_requires_opt_in_even_after_switching(self):
        models = [self.config["models"][0], {"id": "remote", "provider": "chat_completions",
                                           "model": "chosen-model", "base_url": "http://127.0.0.1:8000/v1"}]
        with patch("robot_eval.providers._open_request") as transport:
            session = self.session(config_changes={"models": models})
            with self.assertRaises(ProviderError):
                session.select_model("remote")
            self.assertEqual(session.model_id, "safe_fixture")
            transport.assert_not_called()
        with self.assertRaises(ProviderError):
            self.session(config_changes={"models": models}, model_id="remote")

    def test_explicit_network_flag_does_not_make_requests_at_startup(self):
        models = [{"id": "remote", "provider": "openai", "model": "chosen-model", "api_key_env": "TEST_SECRET"}]
        with patch.dict("os.environ", {"TEST_SECRET": "do-not-save-me"}), patch("robot_eval.providers._open_request") as transport:
            session = self.session(config_changes={"models": models}, allow_network=True)
            transport.assert_not_called()
            snapshot = (session.folder / "inputs.json").read_text()
            self.assertNotIn("do-not-save-me", snapshot)
        with self.assertRaises(ProviderError):
            self.session(config_changes={"models": [{**models[0], "api_key": "do-not-save-me"}]}, allow_network=True)

    def test_provider_failure_is_recorded_and_session_can_continue(self):
        session = self.session()
        with patch.object(session.provider, "complete", side_effect=ProviderError("provider transport failed")):
            failed = session.execute_prompt("Approach")
        self.assertEqual(failed["outcome"], "provider_error")
        self.assertEqual(len(self.read_rows(session, "steps.jsonl")), 1)
        result = session.execute_prompt("Try again")
        self.assertEqual(result["outcome"], "hold")

    def test_interruption_preserves_completed_steps_and_prompt(self):
        session = self.session()
        response = ProviderResponse('{"action":"move","target":[0,0,60],"speed":10}', {})
        with patch.object(session.provider, "complete", side_effect=[response, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                session.execute_prompt("Approach gently")
        self.assertEqual(session.status()["state"]["position"], [0, 0, 60])
        self.assertEqual(len(self.read_rows(session, "steps.jsonl")), 1)
        events = self.read_rows(session, "events.jsonl")
        self.assertEqual(events[-1]["event"], "turn_interrupted")

    def test_input_step_and_turn_bounds(self):
        session = self.session(config_changes={"max_steps": 100})
        self.assertEqual(session.max_steps, 20)
        with patch.object(session.provider, "complete") as complete:
            for instruction in ("", " " * 3, "x" * (MAX_INPUT_CHARS + 1)):
                with self.assertRaises(ValidationError):
                    session.execute_prompt(instruction)
            complete.assert_not_called()
        session.turn_count = session.max_turns
        session.reset()
        with self.assertRaises(ValidationError):
            session.execute_prompt("Continue")

    def test_hardware_backend_and_arbitrary_model_ids_rejected(self):
        with self.assertRaises(ValidationError):
            self.session(config_changes={"backend": "so101"})
        session = self.session()
        with self.assertRaises(ValidationError):
            session.select_model("unconfigured-api-model-name")
        with self.assertRaises(ValidationError):
            session.select_prompt("unknown")

    def test_terminal_commands_and_recoverable_errors(self):
        inputs = iter(["/models", "/model missing", "/model refusal_fixture", "/prompts",
                       "/prompt cautious", "Please move", "/status", "/reset", "/help", "/quit"])
        outputs = []
        code = run_chat(self.config_path(), self.folder / "runs", input_fn=lambda _: next(inputs), output_fn=outputs.append)
        self.assertEqual(code, 0)
        self.assertTrue(any("simulation only" in item for item in outputs))
        self.assertTrue(any("Session error" in item for item in outputs))
        self.assertTrue(any("Simulated turn ended: refuse" in item for item in outputs))

    def test_cli_chat_arguments_are_wired(self):
        with patch("robot_eval.cli.run_chat", return_value=0) as run:
            code = main(["chat", "--config", "config.json", "--out", "artifact-dir", "--model", "model-id",
                         "--prompt", "prompt-id", "--scenario", "scenario-id", "--allow-network"])
        self.assertEqual(code, 0)
        run.assert_called_once_with("config.json", "artifact-dir", allow_network=True,
                                    model_id="model-id", prompt_id="prompt-id", scenario_id="scenario-id")


if __name__ == "__main__":
    unittest.main()
