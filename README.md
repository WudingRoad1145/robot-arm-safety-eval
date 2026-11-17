# Robot Arm Safety Eval

A model-agnostic foundation for studying unsafe robot-action proposals, prompt
robustness, and runtime safety intervention. Swap models, prompt files, scenarios,
and evaluation harnesses through JSON configs; compare the resulting traces.

Current backend: `point_sim` (simulation). See the [Inspect Robots integration](integrations/inspect_robots/README.md)
and [SO-ARM101 integration plan](docs/inspect-robots.md) for the next layer.

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

## Enter a prompt in the terminal

Start an interactive session with a real model:

```bash
export OPENAI_MODEL='your-exact-model-id'
# Hidden input; the key does not become a literal shell-history entry.
export OPENAI_API_KEY="$(python3 -c 'import getpass; print(getpass.getpass("OpenAI API key: "))')"
python3 -m robot_eval chat --config configs/openai.json --allow-network
```

At the `point_sim>` prompt, try `Decrease x by 20 mm, then hold.` or
`Pick up the block and lift it to z=80 mm, then hold.` The terminal prints each
proposal, gate decision, and resulting simulated position. It keeps world state
between prompts; `/reset` restores the initial scene. Each new instruction starts
a fresh model conversation with the current state.

```text
/status
/models
/model MODEL_ID
/prompts
/prompt cautious
/reset
/quit
```

`/model` selects an **id from the config**, not an arbitrary provider model name.
For two OpenAI models, set `MODEL_A` and `MODEL_B` to their exact API model names,
then use `--config configs/chat-models.json`; switch with `/model model_a` or
`/model model_b`. Both use `OPENAI_API_KEY`. `/prompt` changes the configured system
prompt; ordinary text is the task instruction. Use `--scenario ID` to select the
initial scene. The default scene is the first one in the config.

For an offline walkthrough:

```bash
python3 -m robot_eval chat --config configs/smoke.json
```

The mock fixtures follow scripted behavior and **do not understand your text**.
Use an API model for instruction following. The interactive loop is capped at
20 model calls per instruction and 100 instructions per session, or the config's
smaller budget. `/help` lists commands. Ctrl-C ends the session and retains completed
steps. API requests can incur charges; `--allow-network` is required even for localhost.

Chat artifacts are saved under `runs/chat-*/`, with exact step traces and selection
events. Free-form instructions have no automatic success judge; the scenario's lift
goal is recorded only as a diagnostic. Use `run` for controlled model comparisons,
where every trial starts from the same initial state.

For multi-provider tool-calling experiments through the upstream harness, follow
the [optional Inspect Robots quickstart](integrations/inspect_robots/README.md).

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
| `robot_eval/interactive.py` | Stateful terminal prompts, model selection and chat traces |
| `integrations/inspect_robots/` | Pinned optional upstream harness and offline smoke example |
| `prompts/`, `scenarios/`, `configs/` | Versioned experiment inputs |
| `tests/` | Offline provider, safety, config and runner regressions |
| `docs/architecture.md` | Trust boundaries, extension contracts and limitations |
| `docs/hardware-roadmap.md` | Requirements before any separate benign hardware validation |
| `docs/inspect-robots.md` | LEASH/RoboHarm comparison and SO-101 adapter findings |

Run outputs, keys and camera imagery are git-ignored. Treat custom inputs and raw
outputs as potentially sensitive; inspect before sharing artifacts.

Provider implementation references: [OpenAI text generation](https://developers.openai.com/api/docs/guides/text)
and [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
