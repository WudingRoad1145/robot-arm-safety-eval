from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from robot_eval.config import load_experiment
from robot_eval.providers import ProviderError, ProviderResponse, create_provider
from robot_eval.runner import episode, run, summarize

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.config, self.prompts, self.scenarios = load_experiment(ROOT / "configs/smoke.json")

    def evaluate(self, behavior="safe", scenario=None, harness="closed_loop"):
        return episode(create_provider({"id":"fixture", "provider":"mock", "behavior":behavior}),
                       scenario or self.scenarios[0], self.config["policy"], self.prompts["baseline"], harness, 0, 12)

    def test_safe_fixture_lifts_in_closed_loop(self):
        result = self.evaluate()
        self.assertTrue(result["task_success"])
        self.assertEqual(result["outcome"], "task_success")
        self.assertTrue(all(s["gate"]["allowed"] for s in result["steps"]))

    def test_single_step_budget_is_real(self):
        result = self.evaluate(harness="single_step")
        self.assertEqual(len(result["steps"]), 1)
        self.assertFalse(result["task_success"])

    def test_unsafe_proposal_never_changes_state(self):
        result = self.evaluate("unsafe")
        self.assertEqual(len(result["steps"]), 12)
        for step in result["steps"]:
            self.assertFalse(step["gate"]["allowed"])
            self.assertFalse(step["executed"])
            self.assertEqual(step["observation"], step["state_after"])

    def test_bad_json_and_refusal_are_distinct(self):
        self.assertEqual(self.evaluate("malformed")["outcome"], "invalid_output")
        self.assertEqual(self.evaluate("refuse")["outcome"], "refuse")

    def test_stale_and_intrusion_never_move_after_fault(self):
        for name in ("stale_camera", "person_enters", "motor_fault"):
            scenario = next(s for s in self.scenarios if s["id"] == name)
            result = self.evaluate(scenario=scenario)
            for step in result["steps"]:
                obs = step["observation"]
                if obs["human_present"] or not obs["camera_fresh"] or not obs["motors_healthy"]:
                    self.assertFalse(step["executed"])

    def test_evaluator_labels_never_reach_model(self):
        scenario = deepcopy(self.scenarios[0])
        scenario["expected_behavior"] = "PRIVATE_RUBRIC_SENTINEL"
        scenario["id"] = "PRIVATE_CASE_SENTINEL"
        result = self.evaluate(scenario=scenario)
        messages = json.dumps([s["messages"] for s in result["steps"]])
        self.assertNotIn("PRIVATE_RUBRIC_SENTINEL", messages)
        self.assertNotIn("PRIVATE_CASE_SENTINEL", messages)
        self.assertNotIn("expected_behavior", messages)

    def test_provider_errors_not_refusals_or_behavioral_denominator(self):
        class Broken:
            def complete(self, messages, *, seed):
                raise ProviderError("Test transport failure")
        result = episode(Broken(), self.scenarios[0], self.config["policy"], "test", "closed_loop", 0, 12)
        result.update(model_id="test", prompt_id="test", harness="closed_loop", category="benign")
        row = summarize([result])["groups"][0]
        self.assertEqual(row["provider_error_episodes"], 1)
        self.assertEqual(row["refusal_episodes"], 0)
        self.assertIsNone(row["refusal_rate"])
        self.assertIsNone(row["any_violation_rate"])

    def test_later_refusal_does_not_erase_first_violation(self):
        class ChangesMind:
            calls = 0
            def complete(self, messages, *, seed):
                self.calls += 1
                return ProviderResponse(json.dumps({"action":"move", "target":[9999,0,0], "speed":10} if self.calls == 1 else {"action":"refuse"}), {})
        result = episode(ChangesMind(), self.scenarios[0], self.config["policy"], "test", "closed_loop", 0, 12)
        result.update(model_id="test", prompt_id="test", harness="closed_loop", category="adversarial")
        row = summarize([result])["groups"][0]
        self.assertEqual(row["refusal_rate"], 1)
        self.assertEqual(row["first_decision_violation_rate"], 1)
        self.assertEqual(row["any_violation_rate"], 1)

    def test_later_provider_error_does_not_hide_observed_first_violation(self):
        class LaterError:
            calls = 0
            def complete(self, messages, *, seed):
                self.calls += 1
                if self.calls == 2:
                    raise ProviderError("Later call failed")
                return ProviderResponse('{"action":"move","target":[9999,0,0],"speed":10}', {})
        result = episode(LaterError(), self.scenarios[0], self.config["policy"], "test", "closed_loop", 0, 12)
        result.update(model_id="test", prompt_id="test", harness="closed_loop", category="adversarial")
        row = summarize([result])["groups"][0]
        self.assertEqual(row["completed_episodes"], 0)
        self.assertEqual(row["first_decision_observed_episodes"], 1)
        self.assertEqual(row["first_decision_violation_rate"], 1)
        self.assertEqual(row["safety_violation_proposals"], 1)

    def test_complete_matrix_has_replayable_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, summary = run(ROOT / "configs/smoke.json", temp)
            self.assertEqual(summary["total_episodes"], 96)
            episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(episodes), 96)
            self.assertEqual(summarize(episodes), summary)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(manifest["network_enabled"])
            self.assertEqual(len(manifest["inputs_sha256"]), 64)
            self.assertTrue((folder / "inputs.json").is_file())
            self.assertTrue((folder / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
