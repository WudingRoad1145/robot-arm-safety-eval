"""Text-only model adapters; nothing here can access a physical robot.

Mock policies exercise harness plumbing, not model safety. Network calls require
an explicit opt-in, never retry, and never follow redirects. Provider output is
untrusted text for the runner's action parser and independent simulation gate.
"""

from dataclasses import dataclass
import http.client
import ipaddress
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request


MAX_RESPONSE_BYTES = 1_048_576
MAX_REQUEST_BYTES = 1_048_576


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    metadata: dict


class ProviderError(Exception):
    """Sanitized provider failure: no credentials, response bodies, or URLs."""


def _number(value, label, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not minimum <= value <= maximum or not math.isfinite(value)):
        raise ProviderError(f"{label} must be a finite number in [{minimum}, {maximum}]")
    return value


def _messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ProviderError("messages must be a nonempty list")
    result = []
    for message in messages:
        if (not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message.get("role") not in {"system", "developer", "user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise ProviderError("messages must contain text and a supported role")
        result.append({"role": message["role"], "content": message["content"]})
    return result


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProviderError("seed must be an integer")


def _vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ProviderError("mock observation requires three-dimensional positions")
    return [_number(x, "mock coordinate", -1e9, 1e9) for x in value]


def _observation(messages):
    content = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    if content is None:
        raise ProviderError("mock requires a user observation")
    # The runner emits JSON. Accept a prose prefix for custom text harnesses too.
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict) and isinstance(value.get("observation"), dict):
            return value
    raise ProviderError("mock requires a JSON object containing observation")


class MockProvider:
    """Deterministic plumbing fixture, deliberately not a language model."""

    def __init__(self, config):
        self.behavior = config.get("behavior", "safe")
        if not isinstance(self.behavior, str) or self.behavior not in {"safe", "unsafe", "malformed", "refuse"}:
            raise ProviderError("unknown mock behavior")

    def complete(self, messages, *, seed):
        _seed(seed)
        messages = _messages(messages)
        metadata = {"provider": "mock", "synthetic": True,
                    "behavior": self.behavior, "seed": seed}
        if self.behavior == "malformed":
            return ProviderResponse("{not valid JSON", metadata)
        if self.behavior == "unsafe":
            proposal = {"action": "move", "target": [9999, 0, 0], "speed": 10,
                        "reason": "synthetic invalid boundary proposal"}
        elif self.behavior == "refuse":
            proposal = {"action": "refuse", "reason": "synthetic refusal fixture"}
        else:
            data = _observation(messages)
            observation = data["observation"]
            # This planner ignores language entirely. Its success is plumbing
            # evidence only; it cannot measure instruction-following safety.
            if (observation.get("camera_fresh", True) is not True
                    or observation.get("motors_healthy", True) is not True
                    or observation.get("human_present", False) is not False):
                proposal = {"action": "stop", "reason": "synthetic scene precondition failed"}
            else:
                position = _vector(observation.get("position"))
                target = _vector(observation.get("object_position"))
                holding = observation.get("holding")
                if not isinstance(holding, bool):
                    raise ProviderError("mock observation holding must be boolean")
                if holding:
                    goal = observation.get("goal", {})
                    if not isinstance(goal, dict):
                        raise ProviderError("mock goal must be an object")
                    target = position[:]
                    target[2] = _number(goal.get("lift_z", 80), "mock lift height", -1e9, 1e9)
                    if position[2] >= target[2] - 0.01:
                        return ProviderResponse(json.dumps({"action": "hold", "reason": "lift reached"}), metadata)
                delta = [b - a for a, b in zip(position, target)]
                distance = math.sqrt(sum(value * value for value in delta))
                if distance <= 5 and not holding:
                    proposal = {"action": "grasp", "reason": "at simulated object"}
                else:
                    limits = observation.get("limits", {})
                    if not isinstance(limits, dict):
                        raise ProviderError("mock limits must be an object")
                    max_step = _number(limits.get("max_step_mm", 30), "mock step limit", 0.001, 1e9)
                    scale = min(1, min(30, max_step) / distance)
                    proposal = {"action": "move", "target": [a + d * scale for a, d in zip(position, delta)],
                                "speed": 10, "reason": "bounded simulated approach"}
        return ProviderResponse(json.dumps(proposal, allow_nan=False), metadata)


def _endpoint(base_url, suffix):
    if not isinstance(base_url, str) or not base_url or any(ord(c) <= 32 for c in base_url):
        raise ProviderError("base_url must be a valid API URL")
    try:
        parsed = urllib.parse.urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ProviderError("base_url must be a valid API URL") from None
    if (not hostname or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or "?" in base_url or "#" in base_url
            or parsed.path.rstrip("/") != "/v1" or port == 0):
        raise ProviderError("base_url must end in /v1 without credentials, query, or fragment")
    loopback = hostname.lower() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ProviderError("base_url requires HTTPS except for loopback HTTP")
    return base_url.rstrip("/") + suffix


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ProviderError("provider redirect refused")


def _open_request(request, timeout):
    # Disable environment proxy discovery: the configured endpoint is the only
    # destination authorized for this text and these credentials.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    return opener.open(request, timeout=timeout)


def _reject_constant(_):
    raise ValueError("non-finite JSON number")


def _post_json(endpoint, payload, api_key, timeout):
    try:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ProviderError("request is not valid finite JSON") from None
    if len(body) > MAX_REQUEST_BYTES:
        raise ProviderError("provider request exceeds size limit")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key is not None:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with _open_request(request, timeout) as response:
            status = response.status
            if not isinstance(status, int) or not 200 <= status < 300:
                raise ProviderError(f"provider HTTP status {status}" if isinstance(status, int)
                                    else "invalid provider HTTP status")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        raise ProviderError(f"provider HTTP status {code}") from None
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        raise ProviderError("provider transport failed") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError("provider response exceeds size limit")
    try:
        data = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ProviderError("provider returned invalid JSON") from None
    if not isinstance(data, dict):
        raise ProviderError("provider returned a non-object response")
    return data


class NetworkProvider:
    def __init__(self, config, *, allow_network):
        if allow_network is not True:
            raise ProviderError("network providers require explicit allow_network=True")
        self.provider = config["provider"]
        self.model = config.get("model")
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 512:
            raise ProviderError("provider requires an explicit model string")
        self.timeout = _number(config.get("timeout_s", 30), "timeout_s", 1, 120)
        self.temperature = config.get("temperature")
        if self.temperature is not None:
            _number(self.temperature, "temperature", 0, 2)
        self.max_tokens = config.get("max_output_tokens", 1024)
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ProviderError("max_output_tokens must be an integer")
        _number(self.max_tokens, "max_output_tokens", 1, 32768)
        default_base = "https://api.openai.com/v1" if self.provider == "openai" else None
        suffix = "/responses" if self.provider == "openai" else "/chat/completions"
        self.endpoint = _endpoint(config.get("base_url", default_base), suffix)
        default_env = "OPENAI_API_KEY" if self.provider == "openai" else None
        self.api_key_env = config.get("api_key_env", default_env)
        if self.api_key_env is not None and (not isinstance(self.api_key_env, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)):
            raise ProviderError("api_key_env must name an environment variable")
        if self.provider == "openai" and self.api_key_env is None:
            raise ProviderError("openai requires api_key_env")

    def complete(self, messages, *, seed):
        _seed(seed)
        messages = _messages(messages)
        key = None
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env)
            if not key or any(ord(c) <= 32 or ord(c) >= 127 for c in key):
                raise ProviderError("API key environment variable is missing or invalid")
        payload = {"model": self.model}
        if self.provider == "openai":
            payload.update(input=messages, store=False, max_output_tokens=self.max_tokens)
        else:
            payload.update(messages=messages, max_tokens=self.max_tokens, stream=False)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = _post_json(self.endpoint, payload, key, self.timeout)
        metadata = {"provider": self.provider, "model": self.model,
                    "seed_requested": seed, "seed_sent": False,
                    "request_parameters": {k: v for k, v in payload.items()
                                           if k not in {"messages", "input"}}}
        if isinstance(data.get("model"), str):
            metadata["reported_model"] = data["model"]
        # Seeds are not portable across endpoints/models; do not imply API
        # determinism merely because simulation initial states are repeatable.
        if isinstance(data.get("usage"), dict):
            allowed = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"}
            metadata["usage"] = {k: v for k, v in data["usage"].items()
                                 if k in allowed and type(v) is int and v >= 0}
        if self.provider == "openai":
            return self._responses(data, metadata)
        return self._chat(data, metadata)

    @staticmethod
    def _refusal(reason, metadata):
        metadata["native_refusal"] = True
        metadata["refusal_truncated"] = len(reason) > 2000
        return ProviderResponse(json.dumps({"action": "refuse", "reason": reason[:2000]}), metadata)

    def _responses(self, data, metadata):
        if data.get("error") is not None or data.get("status", "completed") != "completed":
            raise ProviderError("provider response did not complete")
        output = data.get("output")
        if not isinstance(output, list):
            raise ProviderError("provider response has no output list")
        texts, refusals = [], []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                raise ProviderError("provider message has invalid content")
            for part in content:
                if not isinstance(part, dict):
                    raise ProviderError("provider message has invalid content")
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                    refusals.append(part["refusal"])
        if refusals:
            return self._refusal("\n".join(refusals), metadata)
        if not texts:
            raise ProviderError("provider response has no text output")
        return ProviderResponse("".join(texts), metadata)

    def _chat(self, data, metadata):
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("provider response has no choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("provider response has invalid message")
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            return self._refusal(refusal, metadata)
        if choice.get("finish_reason") not in {None, "stop"}:
            raise ProviderError("provider response did not complete as text")
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError("provider response has no text output")
        return ProviderResponse(content, metadata)


def create_provider(config: dict, *, allow_network: bool = False):
    if not isinstance(config, dict):
        raise ProviderError("provider configuration must be an object")
    allowed = {"id", "provider", "behavior", "model", "model_env", "base_url",
               "api_key_env", "timeout_s", "max_output_tokens", "temperature"}
    if set(config) - allowed:
        raise ProviderError("unknown provider configuration field; credentials must use api_key_env")
    kind = config.get("provider")
    if not isinstance(kind, str):
        raise ProviderError("provider must be a string")
    if kind == "mock":
        return MockProvider(config)
    if kind in {"openai", "chat_completions"}:
        return NetworkProvider(config, allow_network=allow_network)
    raise ProviderError("unknown provider; choose mock, openai, or chat_completions")
