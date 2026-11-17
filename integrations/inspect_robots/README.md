# Optional Inspect Robots integration

This runs the upstream Inspect Robots harness with its built-in **2D CubePick
reach mock**: a point moves toward another point. It is not this repository's
3D `point_sim`, a grasp/lift benchmark, or an SO-ARM101 digital twin. Scores remain
in separate upstream logs and must not be merged with `robot_eval` safety metrics.
Neither script selects hardware, reads a user rig config, or starts a simulator
launcher. The core `robot_eval` runner remains dependency-free.

## Install and run offline

From this repository root, use a separate environment (Python 3.11+, tested with
3.13). Installation requires network; the smoke itself does not:

```bash
python3.13 -m venv .venv-inspect
.venv-inspect/bin/python -m pip install -r integrations/inspect_robots/requirements.txt
.venv-inspect/bin/python integrations/inspect_robots/smoke.py
.venv-inspect/bin/python -m unittest discover -s integrations/inspect_robots -p 'test_*.py' -v
```

The smoke constructs `CubePickEmbodiment`, `ScriptedPolicy`, `ClampGuard` and
scorers explicitly through the public Python API. It makes no model calls and
requires no keys or robot. Two seeded scenes × two epochs × at most 32 steps
produce four trials. Exit code zero requires four successful reach trials with
no errors; the policy is a scripted oracle, so this is an installation check,
not a model score. Logs go under `runs/inspect_robots/smoke-*` by default.
The wrapper tests use the real upstream agent with an HTTP mock transport and
block socket connections; they make no provider calls.

Both upstream packages are pinned to reviewed commit
[`772019598e42ccc7921a043dee3a2ea2692020ea`](https://github.com/robocurve/inspect-robots/tree/772019598e42ccc7921a043dee3a2ea2692020ea).
Transitive dependencies retain upstream constraints; this is a source pin, not
a complete dependency lock. Save `pip freeze` alongside a serious experiment.

## Run an API model in the same mock

Set the selected provider's documented API-key environment variable externally.
Use the exact model identifier offered by that provider, and an API wire that it
supports. The wrapper intentionally has no default model, no automatic task
generator or model-based grader, and no selectable embodiment:

```bash
export PROVIDER='your-provider-prefix'
export MODEL='your-exact-model-id'
export WIRE='chat'  # or responses/messages, as supported by the provider
.venv-inspect/bin/python integrations/inspect_robots/run_agent.py \
  --model "$PROVIDER/$MODEL" --wire "$WIRE" --allow-network
```

For an explicitly chosen compatible server, add `--base-url "$API_BASE_URL"`
and `--api-key-env "$API_KEY_VARIABLE_NAME"`; that variable's value is the key.
For a server without authentication, name an unset variable. Choose a trusted
endpoint: the selected key and synthetic observations are sent there. Direct
provider routing and its credential variable names are documented in the pinned
[agent plugin](https://github.com/robocurve/inspect-robots/blob/772019598e42ccc7921a043dee3a2ea2692020ea/plugins/inspect-robots-agent/README.md).

Remote calls may incur charges. Defaults are one trial, at most 32 environment
steps and eight model calls, a 30-second per-request timeout, and no retries.
`--max-steps` accepts 1–64 and `--max-calls` accepts 1–12. These are execution
budgets, not a dollar cap. Upstream `status=success` means the evaluation ran;
`success_at_end` separately reports whether the point reached the cube.
Agent logs, synthetic frames, action traces and transcripts go under
`runs/inspect_robots/agent-*`; inspect artifacts before sharing them.

## Compare prompts

At this pinned revision, `LLMAgentPolicy` has **no public `system_prompt`
replacement argument**. Its `reset()` builds an internal system prompt with
tool/embodiment instructions. Supported additions are `eval_notes` (short text)
and `prior_learnings` (a UTF-8 file appended as advisory notes). This wrapper maps
those to `--eval-notes` and `--notes-file`:

```bash
.venv-inspect/bin/python integrations/inspect_robots/run_agent.py \
  --model "$PROVIDER/$MODEL" --wire "$WIRE" --allow-network \
  --notes-file path/to/prompt-variant.txt \
  --eval-notes 'Reach the cube within the available step budget.'
```

Repeat with another notes file while keeping model, seed, budgets and task fixed.
Upstream logs record the notes path/hash and the resulting system prompt in the
policy transcript. This appends instructions; it does not replace the base
system prompt. Full replacement requires a reviewed custom Policy implementation
or an upstream extension, not private-template monkeypatching. See the pinned
[constructor and reset implementation](https://github.com/robocurve/inspect-robots/blob/772019598e42ccc7921a043dee3a2ea2692020ea/plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py).

Porting our scenarios and action/gate contracts into an Inspect Robots
Embodiment/Guard/Scorer adapter is a separate step. Until then, use `robot_eval`
for the existing safety matrix and this integration to explore upstream agent
tools, logs and public extension points.
