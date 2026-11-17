# Robot Arm Safety Eval

A model-agnostic foundation for studying unsafe robot-action proposals, prompt
robustness, and runtime safety intervention. Swap models, prompt files, scenarios,
and evaluation harnesses through JSON configs; compare the resulting traces.

**This version cannot operate a physical arm.** It uses a deterministic abstract
point-gripper simulator. It is not an SO-ARM101 digital twin and does not validate
real-world safety, camera alignment, grasp force, or collision-free arm motion.
The live arm scripts and camera images from the preceding experiment are deliberately
not imported. Adversarial cases belong in simulation.

## Run in one minute

Python 3.10+; no third-party packages, API keys, robot, or network needed:

```bash
python3 -m robot_eval list
make test
make smoke
```

The smoke matrix runs **96 episodes**: four synthetic provider fixtures × two prompts
× two harnesses × six scenarios. Mock fixtures exercise success, blocked actions,
malformed responses and explicit refusal. They are not real-model evaluation results.

Each `runs/<timestamp>-<id>/` contains:

- `inputs.json`: the resolved config, prompt contents and scenario snapshots.
- `manifest.json`: source/config hashes, Git revision, budgets and completion status.
- `episodes.jsonl`: exact messages, raw responses, provider metadata, gate decisions,
  executed simulated actions and before/after state at each step.
- `summary.json` and `report.md`: comparisons split by model, prompt, harness and category.

Regenerate a report without another model call:

```bash
python3 -m robot_eval report runs/YOUR_RUN_DIRECTORY
```

Run from this repository root. Alternatively, `python3 -m pip install -e .` installs
the `robot-eval` command; pass an absolute `--config` path when running elsewhere.
Editable installation requires setuptools; the direct module commands need only Python.

## Switch models

OpenAI Responses API:

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-exact-model-id'
python3 -m robot_eval run --config configs/openai.json --allow-network
```

An already-running OpenAI-compatible server (for example a local serving stack):

```bash
export LOCAL_MODEL='exact-model-name-served-by-your-endpoint'
python3 -m robot_eval run --config configs/local.json --allow-network
```

Edit `base_url` in `configs/local.json` for your server. Add `api_key_env` if it
requires authentication. Endpoint compatibility is server-specific. HTTP is restricted
to loopback; remote endpoints require HTTPS. No requests are made without
`--allow-network`, including requests to localhost. Remote calls can incur charges
and send the configured prompts and synthetic observations to that provider.
The `.env.example` file is not automatically loaded.

For comparisons, put multiple entries in `models`; each entry has a unique `id` and
an exact `model` or a `model_env` naming a shell environment variable. Provider and
sampling settings are captured in artifacts. Do not assume a seed makes hosted model
output deterministic. Live provider calls have not been run as part of the bootstrap.

## Switch prompts and harnesses

Copy `configs/smoke.json` and edit its lists. All file paths resolve relative to the
config file, not the shell's current directory.

```json
{
  "models": [{"id": "model_a", "provider": "openai", "model_env": "OPENAI_MODEL", "api_key_env": "OPENAI_API_KEY"}],
  "prompts": [{"id": "baseline", "path": "../prompts/baseline.txt"}],
  "harnesses": ["single_step", "closed_loop"],
  "seeds": [0, 1, 2],
  "max_steps": 12,
  "max_episodes": 200
}
```

This is an excerpt; retain the version, backend, policy and scenarios from the
complete example. `single_step` measures the first decision only. `closed_loop`
feeds observations and gate feedback back to the model for a bounded episode.
Keep their scores separate: they have different opportunities to finish a task.

The common action contract is appended to all prompt variants. The safety gate
stays unchanged across prompt experiments. To compare harness logic beyond the two
built-ins, extend `runner.episode` and the config allowlist together, with tests.

## What to measure

Model proposals, gate intervention and simulated task outcomes are separate:

- First-decision and any-violation-per-episode rates retain unsafe attempts even when
  a model later refuses or corrects itself.
- Explicit refusal, invalid JSON, infeasible grasps, safety blocks, provider errors
  and task success are different outcomes.
- Benign scenarios measure task utility; adversarial cases test boundary pressure and
  untrusted scene instructions; fault cases test responses to stale observations,
  a simulated person entering, or unhealthy motors.
- First-decision rates use all observed first responses, including trials with a later
  provider error. Other episode rates exclude provider-error trials and count those
  errors separately. Raw proposal counts retain partial traces from failed trials.

Scenario `expected_behavior` is a human annotation, not an implemented semantic
judge. Category/id/rubric labels are never sent to the model. Automated grading in
this foundation covers the explicit action constraints and simulated lift goal only.
No aggregate score should be presented as a real-world safety certification.

## Repository map

| File | Responsibility |
|---|---|
| `robot_eval/providers.py` | Provider protocol, mock fixtures, Responses and compatible Chat Completions adapters |
| `robot_eval/config.py` | Config/scenario validation and model-name resolution |
| `robot_eval/safety.py` | Strict JSON actions and independent deterministic gate |
| `robot_eval/simulation.py` | Abstract point-gripper transitions |
| `robot_eval/runner.py` | Matrices, harnesses, traces, metrics and reports |
| `prompts/`, `scenarios/`, `configs/` | Versioned experiment inputs |
| `tests/` | Offline provider, safety, config and runner regressions |
| `docs/architecture.md` | Trust boundaries, extension contracts and limitations |
| `docs/hardware-roadmap.md` | Requirements before any separate benign hardware validation |

Run outputs, keys and camera imagery are git-ignored. Treat custom inputs and raw
outputs as potentially sensitive; inspect before sharing artifacts.

Provider implementation references: [OpenAI text generation](https://developers.openai.com/api/docs/guides/text)
and [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
