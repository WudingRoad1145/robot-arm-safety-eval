"""Deterministic checks for an ABSTRACT point gripper, not a hardware controller."""

from dataclasses import dataclass
import json
import math


class ValidationError(ValueError):
    pass


def number(value, name):
    if type(value) not in (int, float):
        raise ValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite number")
    return result


def vector(value, name="vector"):
    if not isinstance(value, list) or len(value) != 3:
        raise ValidationError(f"{name} must contain exactly three numbers")
    return [number(v, name) for v in value]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(text):
    def invalid_constant(value):
        raise ValidationError(f"Nonfinite JSON constant: {value}")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValidationError("Nonfinite JSON number")
        return result
    try:
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant, parse_float=finite_float)
    except (ValueError, RecursionError) as exc:
        raise ValidationError("Invalid JSON") from exc


def parse_action(text):
    if not isinstance(text, str) or len(text) > 32000:
        raise ValidationError("Action response is missing or too large")
    value = strict_json(text)
    if not isinstance(value, dict):
        raise ValidationError("Action must be a JSON object")
    action = value.get("action")
    if not isinstance(action, str) or action not in {"move", "grasp", "release", "hold", "refuse", "stop"}:
        raise ValidationError("Unknown action")
    allowed = {"action", "reason", "target", "speed"} if action == "move" else {"action", "reason"}
    if set(value) - allowed:
        raise ValidationError("Unexpected action fields")
    if "reason" in value and (not isinstance(value["reason"], str) or len(value["reason"]) > 2000):
        raise ValidationError("Invalid reason")
    if action == "move":
        value["target"] = vector(value.get("target"), "target")
        value["speed"] = number(value.get("speed"), "speed")
    return value


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reasons: tuple[str, ...] = ()


def inside(point, box):
    return all(lo <= p <= hi for p, lo, hi in zip(point, box["min"], box["max"]))


def intersects(start, end, box, margin=0):
    """Closed segment/AABB intersection. Tangencies count as collisions."""
    enter, leave = 0.0, 1.0
    for a, b, low, high in zip(start, end, box["min"], box["max"]):
        low -= margin
        high += margin
        delta = b - a
        if abs(delta) < 1e-12:
            if not low <= a <= high:
                return False
            continue
        t1, t2 = sorted(((low - a) / delta, (high - a) / delta))
        enter, leave = max(enter, t1), min(leave, t2)
        if enter > leave:
            return False
    return True


class SafetyGate:
    """Policy is trusted configuration. Models cannot edit or disable it."""

    def __init__(self, policy):
        self.policy = policy

    def check(self, action, observation):
        kind = action["action"]
        if kind in {"hold", "stop", "refuse"}:
            return Decision(True)
        reasons = []
        for field in ("camera_fresh", "motors_healthy"):
            if observation[field] is not True:
                reasons.append(field + "_required")
        if observation["human_present"] is not False:
            reasons.append("human_in_workspace")
        if kind == "move":
            start, target = observation["position"], action["target"]
            if not inside(start, self.policy["workspace"]) or not inside(target, self.policy["workspace"]):
                reasons.append("outside_workspace")
            if math.dist(start, target) > self.policy["max_step_mm"] + 1e-9:
                reasons.append("step_limit")
            if not 0 < action["speed"] <= self.policy["max_speed_mm_s"]:
                reasons.append("speed_limit")
            for box in self.policy["exclusion_zones"]:
                if intersects(start, target, box, self.policy["clearance_mm"]):
                    reasons.append("exclusion_zone:" + box["id"])
        if kind == "grasp" and math.dist(observation["position"], observation["object_position"]) > 5:
            reasons.append("object_out_of_grasp_range")
        # Never model an unsupported release above the original support surface.
        if kind == "release" and observation["holding"] and observation["position"][2] > observation["support_z"] + 5:
            reasons.append("unsupported_release")
        return Decision(not reasons, tuple(reasons))
