"""Strict, portable JSON experiment configuration. Paths are config-relative."""

from copy import deepcopy
import os
from pathlib import Path
import re

from .safety import ValidationError, inside, intersects, number, strict_json, vector


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def integer(value, name, low, high):
    require(type(value) is int and low <= value <= high, f"Invalid {name}")
    return value


def load_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def box(value, name):
    require(isinstance(value, dict), f"Invalid {name}")
    lo, hi = vector(value.get("min")), vector(value.get("max"))
    require(all(a < b for a, b in zip(lo, hi)), f"Invalid {name} bounds")


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value), "Invalid id")


def load_experiment(path):
    path = Path(path).resolve()
    config = load_json(path)
    require(isinstance(config, dict), "Config must be an object")
    allowed = {"version", "backend", "models", "prompts", "harnesses", "scenarios", "policy", "seeds", "max_steps", "max_episodes"}
    require(not set(config) - allowed, "Unknown experiment fields")
    require(type(config.get("version")) is int and config["version"] == 1, "Unsupported config version")
    require(config.get("backend") == "point_sim", "Only point_sim is supported; physical execution is unavailable")
    integer(config.get("max_steps"), "max_steps", 1, 100)
    integer(config.get("max_episodes"), "max_episodes", 1, 10000)
    for field in ("models", "prompts", "harnesses", "scenarios", "seeds"):
        require(isinstance(config.get(field), list) and len(config[field]) > 0, f"{field} must be a nonempty list")
    for seed in config["seeds"]:
        integer(seed, "seed", 0, 2**32 - 1)
    require(len(set(config["seeds"])) == len(config["seeds"]), "Duplicate seeds")
    require(all(isinstance(h, str) and h in {"single_step", "closed_loop"} for h in config["harnesses"]), "Unknown harness")
    require(len(set(config["harnesses"])) == len(config["harnesses"]), "Duplicate harnesses")
    for field in ("models", "prompts"):
        ids = []
        for entry in config[field]:
            require(isinstance(entry, dict), f"Invalid {field} entry")
            identifier(entry.get("id"))
            ids.append(entry["id"])
        require(len(ids) == len(set(ids)), f"Duplicate {field} ids")
    for model in config["models"]:
        if "model_env" in model:
            require(isinstance(model["model_env"], str), "Invalid model_env")
            require(bool(os.environ.get(model["model_env"])), f"Set model name environment variable {model['model_env']}")
            model["model"] = os.environ[model.pop("model_env")]
    policy = config.get("policy")
    require(isinstance(policy, dict) and set(policy) == {"workspace", "max_step_mm", "max_speed_mm_s", "clearance_mm", "exclusion_zones"}, "Invalid policy fields")
    box(policy["workspace"], "workspace")
    for key in ("max_step_mm", "max_speed_mm_s"):
        require(0 < number(policy[key], key) <= 1000, f"Invalid {key}")
    require(0 <= number(policy["clearance_mm"], "clearance_mm") <= 100, "Invalid clearance")
    require(isinstance(policy["exclusion_zones"], list), "Invalid exclusion zones")
    for zone in policy["exclusion_zones"]:
        box(zone, "exclusion zone")
        identifier(zone.get("id"))
    prompts = {}
    for entry in config["prompts"]:
        require(set(entry) == {"id", "path"}, "Prompt fields must be id/path")
        require(isinstance(entry["path"], str), "Invalid prompt path")
        prompts[entry["id"]] = (path.parent / entry["path"]).read_text(encoding="utf-8")
    scenarios = []
    for filename in config["scenarios"]:
        require(isinstance(filename, str), "Invalid scenario path")
        scenario = load_json(path.parent / filename)
        validate_scenario(scenario, policy)
        scenarios.append(scenario)
    require(len({s['id'] for s in scenarios}) == len(scenarios), "Duplicate scenario ids")
    count = len(config["models"]) * len(prompts) * len(config["harnesses"]) * len(scenarios) * len(config["seeds"])
    require(count <= config["max_episodes"], "Matrix exceeds max_episodes budget")
    return deepcopy(config), prompts, scenarios


def validate_scenario(scenario, policy):
    require(isinstance(scenario, dict), "Invalid scenario")
    require(not set(scenario) - {"id", "category", "instruction", "untrusted_text", "initial_state", "goal", "events", "expected_behavior"}, "Unknown scenario fields")
    identifier(scenario.get("id"))
    require(scenario.get("category") in {"benign", "adversarial", "fault"}, "Invalid scenario category")
    require(isinstance(scenario.get("instruction"), str), "Invalid instruction")
    require(isinstance(scenario.get("untrusted_text", ""), str), "Invalid untrusted_text")
    state = scenario.get("initial_state")
    require(isinstance(state, dict) and set(state) == {"position", "object_position", "human_present", "camera_fresh", "motors_healthy"}, "Invalid initial_state fields")
    for key in ("position", "object_position"):
        vector(state[key], key)
        require(inside(state[key], policy["workspace"]), f"Initial {key} outside workspace")
    for zone in policy["exclusion_zones"]:
        require(not intersects(state["position"], state["position"], zone, policy["clearance_mm"]), "Initial position in exclusion zone")
    for key in ("human_present", "camera_fresh", "motors_healthy"):
        require(type(state[key]) is bool, f"Invalid {key}")
    goal = scenario.get("goal")
    require(isinstance(goal, dict) and set(goal) == {"kind", "lift_z"} and goal["kind"] == "lift", "Only lift goals are supported")
    height = number(goal["lift_z"], "lift_z")
    require(state["object_position"][2] < height <= policy["workspace"]["max"][2], "Invalid lift height")
    events = scenario.get("events", [])
    require(isinstance(events, list), "Invalid events")
    for event in events:
        require(isinstance(event, dict) and set(event) == {"step", "set"}, "Invalid event")
        integer(event["step"], "event step", 0, 99)
        require(isinstance(event["set"], dict) and bool(event["set"]) and not set(event["set"]) - {"human_present", "camera_fresh", "motors_healthy"}, "Events may only modify environmental flags")
        require(all(type(v) is bool for v in event["set"].values()), "Event flags must be bool")
