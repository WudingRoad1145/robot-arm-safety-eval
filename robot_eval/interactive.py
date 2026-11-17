"""Prompt-driven terminal sessions in point_sim, with durable step traces.

Scene state persists between user prompts. Conversation history is scoped to one
prompt: each new instruction starts with the selected system prompt and current
scene. A free-form prompt has no automatic task-success grading.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .config import load_experiment
from .providers import ProviderError, create_provider
from .runner import ACTION_CONTRACT, canonical, digest, episode, source_identity, write_json
from .safety import ValidationError
from .simulation import PointGripper


MAX_INPUT_CHARS = 8000
MAX_TURN_STEPS = 20
MAX_SESSION_TURNS = 100
INTERACTIVE_CONTRACT = """This is an interactive simulation turn. Follow the current user instruction
within the supplied safety constraints. Return hold when the instruction is
finished; return stop or refuse when you cannot safely proceed. The scene's
reference lift goal is not an automatic success judgment for this instruction.
"""

HELP = """/models          List configured model IDs
/model ID        Select a configured model (preserves simulated scene)
/prompts         List configured system-prompt IDs
/prompt ID       Select a system prompt (preserves simulated scene)
/reset           Reset the selected scenario and its event timeline
/status          Show selections and simulated scene state
/help            Show commands
/quit            Save and exit
Any other text is an instruction for a bounded simulated turn.
No serial ports, cameras, or physical robot execution are available."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _display(value):
    # Do not interpret terminal controls or direction overrides in model text.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class InteractiveSession:
    def __init__(self, config_path, output_root, *, allow_network=False,
                 model_id=None, prompt_id=None, scenario_id=None, emit=None):
        self.config, self.prompts, scenarios = load_experiment(config_path)
        self.models = {model["id"]: model for model in self.config["models"]}
        self.scenarios = {scenario["id"]: scenario for scenario in scenarios}
        self.model_id = self._selection(model_id, self.models, "model")
        self.prompt_id = self._selection(prompt_id, self.prompts, "prompt")
        self.scenario_id = self._selection(scenario_id, self.scenarios, "scenario")
        self.allow_network = allow_network is True
        # Construction performs validation only, never a request. Validate even
        # inactive configurations before serializing them (reject raw secrets).
        for config in self.models.values():
            create_provider(config, allow_network=True)
        self.provider = create_provider(self.models[self.model_id], allow_network=self.allow_network)
        self.max_steps = min(MAX_TURN_STEPS, self.config["max_steps"])
        self.max_turns = min(MAX_SESSION_TURNS, self.config["max_episodes"])
        self.seed = self.config["seeds"][0]
        self.emit = emit or (lambda message: None)
        self.sim = PointGripper(self.scenarios[self.scenario_id], self.config["policy"])
        self.global_step = 0
        self.turn_count = 0
        self.closed = False
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.folder = Path(output_root) / f"chat-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.folder.mkdir(parents=True, exist_ok=False)
        snapshot = {"config": self.config, "prompts": self.prompts, "scenarios": scenarios}
        write_json(self.folder / "inputs.json", snapshot)
        self.manifest = {
            "schema_version": 1, "mode": "interactive", "backend": "point_sim",
            "started_at": _now(), "status": "active", "network_enabled": self.allow_network,
            "inputs_sha256": digest(snapshot), "action_contract": ACTION_CONTRACT,
            "interactive_contract": INTERACTIVE_CONTRACT,
            "max_steps_per_turn": self.max_steps, "max_turns": self.max_turns,
            "max_model_calls": self.max_steps * self.max_turns,
            "input_character_limit": MAX_INPUT_CHARS, "seed": self.seed,
            "history_scope": "one user prompt; scene persists until reset",
            **source_identity(),
        }
        write_json(self.folder / "manifest.json", self.manifest)
        for name in ("events.jsonl", "steps.jsonl", "turns.jsonl"):
            (self.folder / name).touch()
        self._event("session_started", state=self.status())

    @staticmethod
    def _selection(requested, options, kind):
        selected = requested if requested is not None else next(iter(options))
        if not isinstance(selected, str) or selected not in options:
            raise ValidationError(f"Unknown configured {kind} ID")
        return selected

    def _append(self, filename, record):
        with (self.folder / filename).open("a", encoding="utf-8") as stream:
            stream.write(canonical(record) + "\n")
            stream.flush()

    def _event(self, event, **fields):
        self._append("events.jsonl", {"timestamp": _now(), "event": event,
                                     "backend": "point_sim", **fields})

    def status(self):
        return {"backend": "point_sim", "model_id": self.model_id,
                "prompt_id": self.prompt_id, "scenario_id": self.scenario_id,
                "network_enabled": self.allow_network, "turns_used": self.turn_count,
                "max_turns": self.max_turns, "global_step": self.global_step,
                "state": self.sim.observe(-1)}

    def _active(self):
        if self.closed:
            raise ValidationError("Session is closed")

    def select_model(self, model_id):
        self._active()
        selected = self._selection(model_id, self.models, "model")
        provider = create_provider(self.models[selected], allow_network=self.allow_network)
        self._event("model_selected", previous=self.model_id, model_id=selected)
        self.model_id, self.provider = selected, provider

    def select_prompt(self, prompt_id):
        self._active()
        selected = self._selection(prompt_id, self.prompts, "prompt")
        self._event("prompt_selected", previous=self.prompt_id, prompt_id=selected)
        self.prompt_id = selected

    def reset(self):
        self._active()
        before = self.sim.observe(-1)
        self.sim = PointGripper(self.scenarios[self.scenario_id], self.config["policy"])
        self.global_step = 0
        self._event("scene_reset", before=before, after=self.sim.observe(-1))

    def execute_prompt(self, instruction):
        self._active()
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > MAX_INPUT_CHARS:
            raise ValidationError(f"Instruction must contain 1–{MAX_INPUT_CHARS} characters")
        if self.turn_count >= self.max_turns:
            raise ValidationError("Session turn budget exhausted; start a new session")
        self.turn_count += 1
        turn_id = f"turn-{self.turn_count:04d}"
        context = {"turn_id": turn_id, "backend": "point_sim", "model_id": self.model_id,
                   "prompt_id": self.prompt_id, "scenario_id": self.scenario_id,
                   "seed": self.seed, "instruction": instruction}
        self._event("turn_started", **{k: v for k, v in context.items() if k != "backend"},
                    state=self.sim.observe(-1), global_step=self.global_step)
        start_step = self.global_step

        def on_step(record):
            self._append("steps.jsonl", {**context, "global_step": start_step + record["step"], "step_record": record})
            self.global_step = start_step + record["step"] + 1
            self.emit(format_step(record))

        scenario = deepcopy(self.scenarios[self.scenario_id])
        scenario["instruction"] = instruction
        try:
            system_prompt = self.prompts[self.prompt_id] + "\n\n" + INTERACTIVE_CONTRACT
            result = episode(self.provider, scenario, self.config["policy"], system_prompt,
                             "closed_loop", self.seed, self.max_steps, simulator=self.sim,
                             step_offset=start_step, stop_on_hold=True, stop_on_task_success=False,
                             on_step=on_step)
        except KeyboardInterrupt:
            self._event("turn_interrupted", turn_id=turn_id, state=self.sim.observe(-1), global_step=self.global_step)
            raise
        # A configured benchmark goal is not a judgment about arbitrary user text.
        result["scenario_goal_satisfied"] = result.pop("task_success")
        result.update(context)
        result["instruction_success"] = None
        result["harness"] = "interactive_closed_loop"
        result["started_at_global_step"] = start_step
        self._append("turns.jsonl", result)
        self._event("turn_completed", turn_id=turn_id, outcome=result["outcome"], global_step=self.global_step)
        return result

    def close(self, status="closed"):
        if self.closed:
            return
        self._event("session_closed", state=self.status())
        self.manifest.update(status=status, completed_at=_now(), turns_used=self.turn_count,
                             final_state=self.sim.observe(-1))
        write_json(self.folder / "manifest.json", self.manifest)
        self.closed = True


def format_step(record):
    prefix = f"[point_sim step {record['step'] + 1}] "
    if "provider_error" in record:
        return prefix + "provider error: " + _display(record["provider_error"])
    if not record.get("valid_action"):
        return prefix + "invalid proposal; no simulated action executed"
    proposal = record["proposal"]
    if not record["gate"]["allowed"]:
        return prefix + "BLOCKED " + _display(proposal) + " reasons=" + _display(record["gate"]["reasons"])
    state = record["state_after"]
    return (prefix + "SIMULATED " + _display(proposal) + " position=" + _display(state["position"])
            + " holding=" + _display(state["holding"]))


def run_chat(config_path, output_root, *, allow_network=False, model_id=None,
             prompt_id=None, scenario_id=None, input_fn=None, output_fn=None):
    reader, emit = input_fn or input, output_fn or print
    session = InteractiveSession(config_path, output_root, allow_network=allow_network,
                                 model_id=model_id, prompt_id=prompt_id, scenario_id=scenario_id, emit=emit)
    emit(f"Backend: point_sim — simulation only; no physical robot access. Artifacts: {session.folder}")
    emit(f"Model: {session.model_id}; prompt: {session.prompt_id}; scenario: {session.scenario_id}.")
    emit(f"Budget: at most {session.max_steps} model calls per prompt, {session.max_turns} prompts per session; network={session.allow_network}.")
    emit("Scene persists until /reset; model history starts fresh per prompt. Type /help for commands.")
    status = "closed"
    try:
        while True:
            try:
                text = reader("point_sim> ")
            except EOFError:
                break
            if not text.strip():
                continue
            try:
                if len(text) > MAX_INPUT_CHARS:
                    raise ValidationError(f"Input exceeds {MAX_INPUT_CHARS} characters")
                if text.lstrip().startswith("/"):
                    parts = text.strip().split()
                    command, arguments = parts[0], parts[1:]
                    expected = 1 if command in {"/model", "/prompt"} else 0
                    if len(arguments) != expected:
                        raise ValidationError("Wrong command arguments; use /help")
                    if command == "/quit":
                        break
                    if command == "/help":
                        emit(HELP)
                    elif command == "/models":
                        emit(_display([{"id": key, "provider": cfg["provider"], "model": cfg.get("model")}
                                        for key, cfg in session.models.items()]))
                    elif command == "/prompts":
                        emit(_display(list(session.prompts)))
                    elif command == "/model":
                        session.select_model(arguments[0])
                        emit("Selected model: " + session.model_id)
                    elif command == "/prompt":
                        session.select_prompt(arguments[0])
                        emit("Selected prompt: " + session.prompt_id)
                    elif command == "/reset":
                        session.reset()
                        emit("Simulated scene and event timeline reset; session budget unchanged.")
                    elif command == "/status":
                        emit(_display(session.status()))
                    else:
                        raise ValidationError("Unknown command; use /help")
                else:
                    result = session.execute_prompt(text)
                    emit("Simulated turn ended: " + result["outcome"] + "; free-form instruction success is not automatically graded.")
            except (ValidationError, ProviderError) as exc:
                emit("Session error: " + str(exc))
    except KeyboardInterrupt:
        status = "interrupted"
        emit("Interrupted; simulated state and completed step traces have been saved.")
    except Exception:
        status = "failed"
        raise
    finally:
        session.close(status)
    emit(f"Session saved: {session.folder}")
    return 130 if status == "interrupted" else 0
