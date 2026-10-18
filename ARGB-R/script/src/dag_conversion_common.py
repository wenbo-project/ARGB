"""Shared utilities for DAG conversion scripts."""

from __future__ import annotations



import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def call_llm(
    prompt: str,
    args: argparse.Namespace | None = None,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    llm_api: str | None = None,
    thinking: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    if args is not None:
        if base_url is None:
            base_url = getattr(args, "llm_base_url", None)
        if model is None:
            model = getattr(args, "model", None)
        if api_key is None:
            api_key = getattr(args, "api_key", None)
        if temperature is None:
            temperature = getattr(args, "temperature", None)
        if max_tokens is None:
            max_tokens = getattr(args, "max_tokens", None)
        if timeout is None:
            timeout = getattr(args, "llm_timeout", None)
        if llm_api is None:
            llm_api = getattr(args, "llm_api", None)
        if thinking is None:
            thinking = getattr(args, "thinking", None)
        if reasoning_effort is None:
            reasoning_effort = getattr(args, "reasoning_effort", None)

    base_url = base_url or "https://api.deepseek.com"
    model = model or ""
    api_key = api_key or ""
    temperature = 0.1 if temperature is None else temperature
    max_tokens = 0 if max_tokens is None else max_tokens
    timeout = 180 if timeout is None else timeout
    llm_api = llm_api or "chat"
    thinking = thinking or "none"
    reasoning_effort = reasoning_effort or ""

    if llm_api == "chat":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON generator for DAG conversion."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if thinking != "none":
            payload["thinking"] = {"type": thinking}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        endpoint = base_url.rstrip("/") + "/chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
            payload["n_predict"] = max_tokens
        endpoint = base_url.rstrip("/") + "/completions"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        body = body.replace("\r", " ").replace("\n", " ")
        if len(body) > 500:
            body = body[:500].rstrip() + "..."
        raise RuntimeError(f"HTTP Error {exc.code}: {exc.reason}; body={body}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not contain choices")
    first = choices[0]
    if llm_api == "chat":
        content = first.get("message", {}).get("content", "")
    else:
        content = first.get("text", "")
    if not isinstance(content, str):
        raise RuntimeError(f"LLM response content is not a string: {type(content).__name__}")
    if not content.strip():
        finish_reason = first.get("finish_reason")
        raise RuntimeError(f"LLM response content is empty; finish_reason={finish_reason!r}")
    return content


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.replace("\x00", "")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = json.loads(_extract_json_object_text(text))
        except json.JSONDecodeError:
            value = json.loads(_repair_truncated_json_object_text(text))
    if not isinstance(value, dict):
        raise ValueError("LLM output must be a JSON object")
    return value


def validate_dag_shape(dag: dict[str, Any]) -> None:
    for field in ("dag_id", "original_question", "nodes", "edges"):
        if field not in dag:
            raise ValueError(f"Converted DAG is missing {field}")
    nodes = dag["nodes"]
    edges = dag["edges"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Converted DAG must contain nodes")
    if not isinstance(edges, list):
        raise ValueError("Converted DAG edges must be a list")
    node_ids = []
    has_gpu_node = False
    for node in nodes:
        if "node_id" not in node or "prompt" not in node:
            raise ValueError("Each node must include node_id and prompt")
        node.setdefault("node_type", "llm_infer")
        normalized_type = str(node["node_type"]).strip().upper()
        if normalized_type in {"CPU", "GPU"}:
            node["node_type"] = normalized_type
            has_gpu_node = has_gpu_node or normalized_type == "GPU"
            metadata = node.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError("Each CPU/GPU node must include metadata")
            if not metadata.get("operation_type"):
                raise ValueError("Each CPU/GPU node must include metadata.operation_type")
            if not metadata.get("device_reason"):
                raise ValueError("Each CPU/GPU node must include metadata.device_reason")
        node_ids.append(node["node_id"])
    if any(str(node.get("node_type", "")).upper() in {"CPU", "GPU"} for node in nodes) and not has_gpu_node:
        raise ValueError("Autonomous LLM DAG must include at least one GPU node")
    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids):
        raise ValueError("Node ids must be unique")
    for edge in edges:
        if edge.get("source") not in node_id_set or edge.get("target") not in node_id_set:
            raise ValueError(f"Edge references unknown node: {edge}")
    dag.setdefault("topology", "mixed")
    dag.setdefault("question_type", "multi_hop_qa")
    dag.setdefault("gold_answer", "")
    dag.setdefault("hop_count", max(len(nodes) - 1, 1))
    dag.setdefault("metadata", {})
    dag["metadata"].setdefault("conversion", "llm")


def _extract_json_object_text(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object start found", text, 0)

    depth = 0
    in_string = False
    escape = False
    for offset, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]

    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    raise json.JSONDecodeError("No complete JSON object found", text, start)


def _placeholder_string_content(key: str | None) -> str:
    if key == "conversion":
        return "llm_autonomous"
    if key == "node_type":
        return "GPU"
    if key == "operation_type":
        return "reasoning"
    if key == "device_reason":
        return "Truncated response"
    if key == "dag_id":
        return "truncated_dag"
    if key == "source":
        return "unknown_source"
    if key == "source_id":
        return "truncated"
    if key == "original_question":
        return "truncated"
    if key == "gold_answer":
        return "truncated"
    if key == "prompt":
        return "truncated"
    return "truncated"


def _placeholder_json_literal(key: str | None) -> str:
    if key in {"hop_count", "source_index", "data_volume_bytes"}:
        return "0"
    return json.dumps(_placeholder_string_content(key), ensure_ascii=False)


def _repair_truncated_json_object_text(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object start found", text, 0)

    text = text[start:]
    contexts: list[dict[str, Any]] = []
    in_string = False
    escape = False
    string_chars: list[str] = []

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
                string_chars.append(char)
                continue
            if char == "\\":
                escape = True
                string_chars.append(char)
                continue
            if char == '"':
                in_string = False
                string_value = "".join(string_chars)
                string_chars = []
                next_index = index + 1
                while next_index < len(text) and text[next_index].isspace():
                    next_index += 1
                next_char = text[next_index] if next_index < len(text) else None
                if contexts and contexts[-1].get("type") == "object" and next_char == ":":
                    contexts[-1]["key"] = string_value
                continue
            string_chars.append(char)
            continue

        if char == '"':
            in_string = True
            escape = False
            string_chars = []
            continue
        if char == "{":
            contexts.append({"type": "object", "key": None})
            continue
        if char == "[":
            contexts.append({"type": "array"})
            continue
        if char == "}":
            if contexts and contexts[-1].get("type") == "object":
                contexts.pop()
            continue
        if char == "]":
            if contexts and contexts[-1].get("type") == "array":
                contexts.pop()
            continue
        if char == ",":
            if contexts and contexts[-1].get("type") == "object":
                contexts[-1]["key"] = None
            continue

    repaired = text.rstrip()
    if in_string:
        key = contexts[-1].get("key") if contexts and contexts[-1].get("type") == "object" else None
        repaired += _placeholder_string_content(key)
        repaired += '"'
    elif contexts:
        key = contexts[-1].get("key") if contexts and contexts[-1].get("type") == "object" else None
        if key is not None and repaired.endswith(":"):
            repaired += " " + _placeholder_json_literal(key)

    while contexts:
        repaired = repaired.rstrip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        ctx = contexts.pop()
        repaired += "}" if ctx.get("type") == "object" else "]"

    return repaired


__all__ = ["call_llm", "load_dotenv", "parse_json_object", "validate_dag_shape"]
