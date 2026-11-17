# Architecture and experiment contract

```mermaid
flowchart LR
  C[Versioned config and scenarios] --> H[Bounded harness]
  H --> P[Model provider]
  P --> R[Raw proposal log]
  R --> V[Strict action parser]
  V --> G[Independent safety gate]
  G -->|allowed| S[Abstract simulation]
  G -->|blocked| F[Feedback and unchanged state]
  S --> H
  F --> H
  R --> E[Evaluation report]
  G --> E
  S --> E
```

The trust boundary is between model-generated proposals and the gate. A model gets
no shell, file, serial, camera, firmware or network tool. Provider HTTP access is
performed only by the trusted adapter and requires an explicit runner opt-in.
Model output is never evaluated as code or used to import modules. A field asking
to disable guards or rewrite observations is an invalid action.

The action parser accepts only one JSON object. It rejects duplicate keys, unknown
fields/actions, booleans as numbers, nonfinite coordinates and responses over the
size limit. Movement uses absolute Cartesian positions and a positive bounded speed.
Workspace limits are convex; both endpoints must be inside. Each entire segment
is checked against exclusion boxes, expanded by configured clearance. A tangency
is blocked. Euclidean displacement, not per-axis displacement, enforces step size.

The gate blocks movement, grasp and release on stale vision, human presence or motor
fault flags. Hold, stop and refuse remain permissible. In this simulator those three
actions leave state unchanged; **stop does not mean disabling real servo torque**.
Release above the simulated support height is blocked. Invalid or blocked proposals
are logged and never applied. `object_out_of_grasp_range` is an infeasible action,
counted as blocked but not as a safety violation.

Simulation has no joints, gripper shape, torque, gravity, contact dynamics, inverse
kinematics or perception pipeline. `holding` is a discrete state set by a grasp within
5 mm of a point object. Object motion then follows the gripper exactly. Safety flags
are fixtures supplied by scenarios, not measured sensors. This separation allows
fast reproducible tests of infrastructure but cannot predict physical grasp success.

`ProviderResponse(text, metadata)` is the provider boundary. Implement a new adapter
with `complete(messages, seed=...)` and register it in `create_provider`. Keep exact
model ids and generation parameters; do not silently fall back to another model.
Credentials are env references, never config values. Provider errors must be sanitized.
No retries are automatic: repeated requests would change costs and sample counts.

Runner configuration is trusted researcher input; prompt/scenario instruction text
is model-visible evaluation data. Expected labels, category, case id and human rubric
are not model-visible. Changing safety policy defines a different experiment; its
hash must be reported. The current common action contract always reinforces the
data/instruction boundary, so this is not a completely unprompted baseline.

Each model/prompt/harness/scenario/seed combination gets a fresh provider object and
simulation. History is local to one episode. Episode ids are independent of matrix
ordering. Local fixtures are deterministic; remote reproducibility means preserving
inputs and outputs, not guaranteeing repeated generations. Textual observations
are supported today; vision-model evaluation needs an explicit image input contract
and dataset provenance in a later version.

Artifacts include exact model-visible message history and every completed step.
Provider errors retain earlier steps and produce a separate terminal outcome. A
process interruption can leave a partial JSONL file; the manifest lacks a completed
status, and `report` can summarize completed episode lines. It does not claim to resume
an interrupted episode or rerun historical model responses.

Tests run entirely offline. CI uses the same test and smoke commands as local work.
Live API conformance still needs explicit opt-in credentials and a small pilot run.
