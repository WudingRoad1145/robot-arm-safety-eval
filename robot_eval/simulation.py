"""Deterministic point-gripper state transitions; no dynamics, images, or SO101 IK."""

from copy import deepcopy
import math


class PointGripper:
    def __init__(self, scenario, policy):
        self.scenario = deepcopy(scenario)
        self.state = deepcopy(scenario["initial_state"])
        self.state["holding"] = False
        self.state["support_z"] = self.state["object_position"][2]
        self.state["goal"] = deepcopy(scenario["goal"])
        self.state["limits"] = deepcopy(policy)
        self.state["untrusted_text"] = scenario.get("untrusted_text", "")

    def observe(self, step):
        for event in self.scenario.get("events", []):
            if event["step"] == step:
                self.state.update(deepcopy(event["set"]))
        return deepcopy(self.state)

    def apply(self, action):
        kind = action["action"]
        if kind == "move":
            self.state["position"] = list(action["target"])
            if self.state["holding"]:
                self.state["object_position"] = list(action["target"])
        elif kind == "grasp":
            if math.dist(self.state["position"], self.state["object_position"]) <= 5:
                self.state["holding"] = True
        elif kind == "release":
            self.state["holding"] = False

    def task_success(self):
        goal = self.scenario["goal"]
        return bool(goal["kind"] == "lift" and self.state["holding"] and self.state["position"][2] >= goal["lift_z"])
