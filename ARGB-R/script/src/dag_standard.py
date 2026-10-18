"""Helpers for normalizing converted DAGs and estimating resource usage."""

from __future__ import annotations



import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any


GPU_NODE_TYPES = {
    "llm_infer",
    "retrieval",
    "aggregate",
    "condition",
    "backtrack",
    "custom",
    "gpu",
}
CPU_NODE_TYPES = {"tool_call", "cpu"}

LOCAL_SIM_CPU_PROFILE = "local_sim_v2"


@dataclass(frozen=True)
class CpuDurationRule:
    """Deterministic CPU duration rule for the local simulation profile."""

    base_ms: int
    chars_per_ms: int
    min_ms: int
    max_ms: int


CPU_DURATION_RULES = {
    "formatting": CpuDurationRule(1, 500, 1, 5),
    "validation": CpuDurationRule(2, 400, 2, 8),
    "comparison": CpuDurationRule(3, 250, 3, 12),
    "aggregation": CpuDurationRule(4, 200, 4, 16),
    "planning": CpuDurationRule(5, 150, 5, 20),
    "retrieval": CpuDurationRule(10, 100, 10, 50),
    "tool_call": CpuDurationRule(15, 50, 15, 100),
    "action": CpuDurationRule(20, 50, 20, 120),
    "computation": CpuDurationRule(4, 150, 4, 30),
    "reasoning": CpuDurationRule(5, 150, 5, 25),
    "default": CpuDurationRule(3, 250, 3, 20),
}

CPU_OPERATION_KEYWORDS = {
    "retrieval": ("retriev", "evidence", "entity_resolution", "lookup", "search"),
    "planning": ("plan", "routing", "route"),
    "reasoning": ("reason",),
    "computation": ("comput", "arithmetic", "calculation", "date_", "height_extraction"),
    "aggregation": ("aggregat", "synthes", "merge"),
    "comparison": ("compar", "decision", "temporal"),
    "validation": ("validat", "check", "select", "boolean", "yes_no"),
    "formatting": ("format", "normaliz", "conversion", "finish", "output"),
}


def standardize_dags(dags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize converted DAG nodes to CPU/GPU while preserving prompts."""
    return [standardize_dag(dag) for dag in dags]


def build_resource_estimates(
    dags: list[dict[str, Any]],
    *,
    cpu_estimation_profile: str | None = None,
    include_node_context: bool = True,
) -> dict[str, Any]:
    """Build a sidecar resource-estimate payload without modifying DAG content."""
    if cpu_estimation_profile not in {None, LOCAL_SIM_CPU_PROFILE}:
        raise ValueError(f"unsupported CPU estimation profile: {cpu_estimation_profile}")

    estimates = [
        _estimate_dag_resources(
            dag,
            cpu_estimation_profile=cpu_estimation_profile,
            include_node_context=include_node_context,
        )
        for dag in dags
    ]
    metadata: dict[str, Any] = {
        "schema": "wenbo_resource_estimates_v1",
        "estimation_method": (
            "operation_aware_local_simulation"
            if cpu_estimation_profile == LOCAL_SIM_CPU_PROFILE
            else "heuristic_from_prompt_and_metadata"
        ),
        "dag_count": len(dags),
        "total_nodes": sum(item["graph_stats"]["num_nodes"] for item in estimates),
        "total_edges": sum(item["graph_stats"]["num_edges"] for item in estimates),
        "node_context_included": include_node_context,
    }
    if cpu_estimation_profile == LOCAL_SIM_CPU_PROFILE:
        metadata.update(
            {
                "cpu_estimation_profile": LOCAL_SIM_CPU_PROFILE,
                "cpu_duration_unit": "ms",
                "cpu_duration_semantics": (
                    "Local simulated operation occupancy; external network and service "
                    "latency are not modeled."
                ),
                "explicit_cpu_policy": (
                    "Recalibrate heuristic estimates; preserve values marked measured or observed."
                ),
                "cpu_duration_rules": {
                    name: asdict(rule) for name, rule in CPU_DURATION_RULES.items()
                },
            }
        )
    return {
        "metadata": metadata,
        "dags": estimates,
    }


def resource_estimate_path(output_path: Any) -> Any:
    return output_path.with_name(f"{output_path.stem}_resource_estimates{output_path.suffix}")


def standardize_dag(dag: dict[str, Any]) -> dict[str, Any]:
    nodes = [_standardize_node(node) for node in dag.get("nodes", [])]
    edges = [_standardize_edge(edge, index) for index, edge in enumerate(dag.get("edges", []))]

    total_tokens = sum(
        int(node.get("tokens", {}).get("tokens", 0))
        for node in nodes
        if isinstance(node.get("tokens"), dict)
    )
    total_data_volume = sum(int(edge.get("data_volume_bytes", 0) or 0) for edge in edges)

    result = dict(dag)
    result["nodes"] = nodes
    result["edges"] = edges
    result.setdefault("dag_type", _infer_dag_type(result))
    result["estimated_cost"] = {
        **(result.get("estimated_cost") if isinstance(result.get("estimated_cost"), dict) else {}),
        "total_tokens": total_tokens,
        "total_data_volume_bytes": total_data_volume,
    }
    result.setdefault("graph_stats", _compute_graph_stats(nodes, edges))
    return result


def _estimate_dag_resources(
    dag: dict[str, Any],
    *,
    cpu_estimation_profile: str | None = None,
    include_node_context: bool = True,
) -> dict[str, Any]:
    nodes = [
        _estimate_node_resource(
            node,
            cpu_estimation_profile=cpu_estimation_profile,
            include_node_context=include_node_context,
        )
        for node in dag.get("nodes", [])
    ]
    edges = [_standardize_edge(edge, index) for index, edge in enumerate(dag.get("edges", []))]
    total_tokens = sum(
        int(node.get("tokens", {}).get("tokens", 0))
        for node in nodes
        if isinstance(node.get("tokens"), dict)
    )
    total_cpu_time_ms = sum(
        int(node.get("cpu", {}).get("time_ms", 0))
        for node in nodes
        if isinstance(node.get("cpu"), dict)
    )
    total_data_volume = sum(int(edge.get("data_volume_bytes", 0) or 0) for edge in edges)
    return {
        "dag_id": dag.get("dag_id", ""),
        "dag_type": dag.get("dag_type", _infer_dag_type(dag)),
        "nodes": nodes,
        "edges": edges,
        "estimated_cost": {
            "total_tokens": total_tokens,
            "total_cpu_time_ms": total_cpu_time_ms,
            "total_data_volume_bytes": total_data_volume,
        },
        "graph_stats": _compute_graph_stats(dag.get("nodes", []), dag.get("edges", [])),
    }


def _standardize_node(
    node: dict[str, Any],
    *,
    cpu_estimation_profile: str | None = None,
    include_node_context: bool = True,
) -> dict[str, Any]:
    raw_type = str(node.get("node_type", "")).strip().lower()
    prompt = str(node.get("prompt", ""))
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}

    result: dict[str, Any] = {"node_id": node["node_id"]}
    if include_node_context and "prompt" in node:
        result["prompt"] = prompt
    if include_node_context and (metadata or raw_type not in {"cpu", "gpu", ""}):
        result["metadata"] = dict(metadata)
        if raw_type not in {"cpu", "gpu", ""}:
            result["metadata"].setdefault("original_node_type", raw_type)
    if raw_type in CPU_NODE_TYPES:
        result["node_type"] = "CPU"
        if cpu_estimation_profile == LOCAL_SIM_CPU_PROFILE and not _has_measured_cpu_time(node):
            result["cpu"] = _estimate_local_sim_cpu(node, raw_type, prompt, metadata)
        else:
            result["cpu"] = _copy_cpu_estimate(node) or {
                "time_ms": _estimate_cpu_time_ms(prompt, metadata)
            }
    else:
        result["node_type"] = "GPU"
        result["tokens"] = _copy_token_estimate(node) or {"tokens": _estimate_tokens(prompt, raw_type)}
    return result


def _estimate_node_resource(
    node: dict[str, Any],
    *,
    cpu_estimation_profile: str | None = None,
    include_node_context: bool = True,
) -> dict[str, Any]:
    standard = _standardize_node(
        node,
        cpu_estimation_profile=cpu_estimation_profile,
        include_node_context=include_node_context,
    )
    return standard


def _standardize_edge(edge: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "edge_id": edge.get("edge_id", f"E{index}"),
        "source": edge["source"],
        "target": edge["target"],
        "data_volume_bytes": int(edge.get("data_volume_bytes", 0) or _estimate_data_volume(edge)),
    }


def _estimate_tokens(prompt: str, raw_type: str) -> int:
    words = len(re.findall(r"\S+", prompt))
    chars = len(prompt)
    base = max(words * 8, chars // 3, 64)
    if raw_type == "aggregate":
        base = int(base * 1.2)
    elif raw_type == "retrieval":
        base = int(base * 0.8)
    return max(base, 32)


def _copy_token_estimate(node: dict[str, Any]) -> dict[str, int] | None:
    tokens = node.get("tokens")
    if isinstance(tokens, dict) and "tokens" in tokens:
        return {"tokens": int(tokens["tokens"] or 0)}
    if "estimated_tokens" in node:
        return {"tokens": int(node["estimated_tokens"] or 0)}
    return None


def _copy_cpu_estimate(node: dict[str, Any]) -> dict[str, Any] | None:
    cpu = node.get("cpu")
    if isinstance(cpu, dict) and "time_ms" in cpu:
        copied: dict[str, Any] = {"time_ms": int(cpu["time_ms"] or 0)}
        for key in ("source", "profile", "operation_class", "payload_chars"):
            if key in cpu:
                copied[key] = cpu[key]
        return copied
    return None


def _estimate_cpu_time_ms(prompt: str, metadata: dict[str, Any]) -> int:
    if "action_input" in metadata:
        size = len(str(metadata.get("action_input", "")))
    elif "action_repr" in metadata:
        size = len(str(metadata.get("action_repr", "")))
    else:
        size = len(prompt)
    return max(50, min(5000, 100 + size * 2))


def _estimate_local_sim_cpu(
    node: dict[str, Any],
    raw_type: str,
    prompt: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    operation_class = _classify_cpu_operation(raw_type, metadata)
    payload_chars = _cpu_payload_chars(operation_class, prompt, metadata)
    rule = CPU_DURATION_RULES[operation_class]
    time_ms = rule.base_ms + math.ceil(payload_chars / rule.chars_per_ms)
    time_ms = max(rule.min_ms, min(rule.max_ms, time_ms))
    return {
        "time_ms": time_ms,
        "source": "estimated",
        "profile": LOCAL_SIM_CPU_PROFILE,
        "operation_class": operation_class,
        "payload_chars": payload_chars,
    }


def _classify_cpu_operation(raw_type: str, metadata: dict[str, Any]) -> str:
    labels = " ".join(
        str(value).strip().lower()
        for value in (
            raw_type,
            metadata.get("operation_type", ""),
            metadata.get("original_node_type", ""),
        )
        if value
    )
    if (
        "tool_call" in labels
        or metadata.get("action_name")
        or "arguments" in metadata
        or "action_input" in metadata
        or "action_repr" in metadata
    ):
        return "tool_call"
    if "action" in labels:
        return "action"
    for operation_class, keywords in CPU_OPERATION_KEYWORDS.items():
        if any(keyword in labels for keyword in keywords):
            return operation_class
    return "default"


def _cpu_payload_chars(
    operation_class: str,
    prompt: str,
    metadata: dict[str, Any],
) -> int:
    if operation_class != "tool_call":
        return len(prompt)

    payload_parts = []
    action_name = metadata.get("action_name")
    if action_name:
        payload_parts.append(str(action_name))
    for key in ("action_input", "action_repr", "arguments"):
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, (dict, list)):
            payload_parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        else:
            payload_parts.append(str(value))
    return len(" ".join(payload_parts)) if payload_parts else len(prompt)


def _has_measured_cpu_time(node: dict[str, Any]) -> bool:
    cpu = node.get("cpu") if isinstance(node.get("cpu"), dict) else {}
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    sources = {
        str(cpu.get("source", "")).strip().lower(),
        str(metadata.get("cpu_time_source", "")).strip().lower(),
        str(metadata.get("resource_estimate_source", "")).strip().lower(),
    }
    return bool(sources & {"measured", "observed", "benchmark"}) and "time_ms" in cpu


def _estimate_data_volume(edge: dict[str, Any]) -> int:
    metadata = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
    mapping = edge.get("data_mapping") if isinstance(edge.get("data_mapping"), dict) else {}
    return max(128, len(str(metadata)) + len(str(mapping)))


def _infer_dag_type(dag: dict[str, Any]) -> str:
    question_type = str(dag.get("question_type", "")).lower()
    if "api" in question_type or "tool" in question_type or "web" in question_type:
        return "tool"
    return "reasoning"


def _compute_graph_stats(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    node_ids = {node["node_id"] for node in nodes}
    in_degree = {node_id: 0 for node_id in node_ids}
    out_degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in out_degree:
            out_degree[source] += 1
        if target in in_degree:
            in_degree[target] += 1
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_roots": sum(1 for degree in in_degree.values() if degree == 0),
        "num_sinks": sum(1 for degree in out_degree.values() if degree == 0),
    }
