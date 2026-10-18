"""Convert ALFWorld trajectory JSON into ARGB DAG JSON.

The converter uses the official ALFWorld trajectory metadata directly:
task description, PDDL parameters, high-level expert actions, and annotations.
It builds a mixed DAG with parallel planning branches followed by a sequential
action chain, which matches the task structure well for Wenbo scheduling.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import time
import urllib.error
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .dag_conversion_common import call_llm, parse_json_object
from .dag_standard import build_resource_estimates, resource_estimate_path, standardize_dags


NON_RETRYABLE_HTTP_CODES = {400, 401, 402, 403, 404, 405, 413, 422}


class FatalLLMError(RuntimeError):
    """An LLM error that retrying with the same configuration cannot fix."""


def clean_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def truncate_text(value: str, max_chars: int) -> str:
    value = clean_text(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def resolve_split_root(input_root: Path, split: str) -> Path:
    direct = input_root / split
    if direct.exists():
        return direct
    if input_root.name == split and input_root.exists():
        return input_root
    raise FileNotFoundError(f"Could not find split '{split}' under {input_root}")


def collect_candidates(input_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for split in splits:
        split_root = resolve_split_root(input_root, split)
        for traj_path in sorted(split_root.rglob("traj_data.json")):
            rel = traj_path.relative_to(split_root)
            if not rel.parts:
                continue
            task_instance = rel.parts[0]
            task_type = task_instance.split("-", 1)[0]
            candidates.append(
                {
                    "path": traj_path,
                    "split": split,
                    "task_instance": task_instance,
                    "task_type": task_type,
                }
            )
    return candidates


def order_candidates(candidates: list[dict[str, Any]], sample_mode: str) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    if sample_mode == "first":
        ordered = sorted(candidates, key=lambda item: str(item["path"]))
    else:
        groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for candidate in sorted(candidates, key=lambda item: str(item["path"])):
            groups[str(candidate["task_type"])].append(candidate)
        task_types = sorted(groups)
        while True:
            progressed = False
            for task_type in task_types:
                group = groups[task_type]
                if not group:
                    continue
                ordered.append(group.popleft())
                progressed = True
            if not progressed:
                break

    for index, candidate in enumerate(ordered):
        candidate["ordered_index"] = index
    return ordered


def choose_annotation(data: dict[str, Any]) -> dict[str, Any]:
    annotations = (((data.get("turk_annotations") or {}).get("anns") or []))
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get("task_desc") or ann.get("high_descs"):
                return ann
        if annotations and isinstance(annotations[0], dict):
            return annotations[0]
    return {}


def extract_high_descs(annotation: dict[str, Any]) -> list[str]:
    descriptions = annotation.get("high_descs") or []
    if not isinstance(descriptions, list):
        return []
    return [clean_text(item) for item in descriptions if clean_text(item)]


def extract_pddl_params(data: dict[str, Any]) -> dict[str, Any]:
    params = data.get("pddl_params")
    return params if isinstance(params, dict) else {}


def extract_scene_info(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene")
    return scene if isinstance(scene, dict) else {}


def is_terminal_step(step: dict[str, Any]) -> bool:
    discrete = step.get("discrete_action") if isinstance(step.get("discrete_action"), dict) else {}
    planner = step.get("planner_action") if isinstance(step.get("planner_action"), dict) else {}
    discrete_action = clean_text(discrete.get("action", "")).lower()
    planner_action = clean_text(planner.get("action", "")).lower()
    return discrete_action == "noop" or planner_action == "end"


def normalize_action_args(args: Any) -> str:
    if isinstance(args, list):
        parts = [clean_text(item) for item in args if clean_text(item)]
        return ", ".join(parts)
    if isinstance(args, dict):
        parts = []
        for key, value in args.items():
            if value in ("", None, [], {}, False):
                continue
            if isinstance(value, list):
                rendered = "[" + ", ".join(clean_text(item) for item in value if clean_text(item)) + "]"
            else:
                rendered = clean_text(value)
            parts.append(f"{key}={rendered}")
        return ", ".join(parts)
    if args in ("", None):
        return ""
    return clean_text(args)


def summarize_action(step: dict[str, Any]) -> str:
    discrete = step.get("discrete_action") if isinstance(step.get("discrete_action"), dict) else {}
    action = clean_text(discrete.get("action", "")) or "UnknownAction"
    args = normalize_action_args(discrete.get("args"))
    if args:
        return f"{action}({args})"
    return action


def branch_specs(pddl_params: dict[str, Any]) -> list[dict[str, str]]:
    branches: list[dict[str, str]] = []
    object_target = clean_text(pddl_params.get("object_target", ""))
    parent_target = clean_text(pddl_params.get("parent_target", ""))
    mrecep_target = clean_text(pddl_params.get("mrecep_target", ""))
    toggle_target = clean_text(pddl_params.get("toggle_target", ""))
    object_sliced = bool(pddl_params.get("object_sliced", False))

    if object_target:
        branches.append(
            {
                "key": "object_target",
                "label": "Identify the target object",
                "value": object_target,
            }
        )
    if parent_target:
        branches.append(
            {
                "key": "parent_target",
                "label": "Identify the destination receptacle",
                "value": parent_target,
            }
        )
    if mrecep_target:
        branches.append(
            {
                "key": "mrecep_target",
                "label": "Identify the movable receptacle",
                "value": mrecep_target,
            }
        )
    if toggle_target:
        branches.append(
            {
                "key": "toggle_target",
                "label": "Identify the auxiliary toggle target",
                "value": toggle_target,
            }
        )
    if object_sliced:
        branches.append(
            {
                "key": "object_sliced",
                "label": "Confirm the slicing requirement",
                "value": "true",
            }
        )

    return branches


def action_steps(data: dict[str, Any], annotation: dict[str, Any]) -> list[dict[str, Any]]:
    high_steps = data.get("plan", {}).get("high_pddl") if isinstance(data.get("plan"), dict) else []
    if not isinstance(high_steps, list):
        high_steps = []
    high_descs = extract_high_descs(annotation)
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(high_steps):
        if not isinstance(step, dict) or is_terminal_step(step):
            continue
        high_idx = step.get("high_idx", index)
        description = ""
        if isinstance(high_idx, int) and high_idx < len(high_descs):
            description = high_descs[high_idx]
        elif index < len(high_descs):
            description = high_descs[index]
        else:
            description = summarize_action(step)
        steps.append(
            {
                "high_idx": high_idx if isinstance(high_idx, int) else index,
                "description": description,
                "action_repr": summarize_action(step),
                "planner_action": step.get("planner_action") if isinstance(step.get("planner_action"), dict) else {},
                "discrete_action": step.get("discrete_action") if isinstance(step.get("discrete_action"), dict) else {},
            }
        )
    return steps


def compact_action_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for step in steps:
        discrete = step.get("discrete_action") if isinstance(step.get("discrete_action"), dict) else {}
        planner = step.get("planner_action") if isinstance(step.get("planner_action"), dict) else {}
        compact.append(
            {
                "high_idx": step["high_idx"],
                "description": step["description"],
                "action_repr": step["action_repr"],
                "discrete_action": clean_text(discrete.get("action", "")),
                "planner_action": clean_text(planner.get("action", "")),
            }
        )
    return compact


def build_llm_prompt(candidate: dict[str, Any], data: dict[str, Any], annotation: dict[str, Any]) -> str:
    task_type = clean_text(data.get("task_type", candidate["task_type"]))
    task_desc = clean_text(annotation.get("task_desc", "")) or f"ALFWorld task: {task_type}"
    pddl_params = extract_pddl_params(data)
    scene_info = extract_scene_info(data)
    branches = branch_specs(pddl_params)
    actions = compact_action_steps(action_steps(data, annotation))
    compact = {
        "source_index": candidate["ordered_index"],
        "split": candidate["split"],
        "task_type": task_type,
        "task_instance": candidate["task_instance"],
        "task_desc": task_desc,
        "scene": {
            "scene_num": scene_info.get("scene_num"),
            "floor_plan": scene_info.get("floor_plan"),
        },
        "pddl_params": pddl_params,
        "candidate_branches": branches,
        "high_level_actions": actions,
    }
    schema = {
        "dag_id": f"alfworld_{candidate['ordered_index'] + 1:04d}",
        "topology": "linear|parallel|diamond|mixed",
        "question_type": "embodied_task",
        "original_question": task_desc,
        "gold_answer": task_desc,
        "nodes": [
            {
                "node_id": "N0",
                "node_type": "CPU|GPU",
                "prompt": "clear executable prompt",
                "metadata": {
                    "original_node_type": "task_planning|constraint_extraction|plan_merge|high_level_action|validation",
                },
            }
        ],
        "edges": [{"source": "N0", "target": "N1"}],
        "hop_count": 1,
        "metadata": {
            "source": "ALFWorld",
            "split": candidate["split"],
            "task_type": task_type,
            "task_instance": candidate["task_instance"],
            "source_index": candidate["ordered_index"],
            "conversion": "llm",
        },
    }
    return (
        "You convert one ALFWorld trajectory into compact ARGB DAG JSON.\n"
        "The DAG should model the real planning and execution workflow of the task.\n"
        "Use the given candidate_branches as parallel planning branches when they are independent.\n"
        "Use GPU for task understanding, branch extraction, plan synthesis, and final validation.\n"
        "Use CPU for concrete environment actions such as navigation, pickup, toggle, place, "
        "clean, heat, cool, or other execution steps.\n"
        "Preserve the expert high-level action order, but do not turn the whole trajectory into "
        "a trivial chain if there are independent planning branches.\n"
        "Return exactly one valid JSON object and no Markdown.\n"
        "Rules:\n"
        "- Node ids must be N0, N1, N2, ... with no gaps.\n"
        "- Node types must be exactly CPU or GPU.\n"
        "- Add only real data/control dependencies.\n"
        "- Prefer a mixed DAG with one planning node, parallel constraint branches, a merge node, "
        "a sequential action chain, and a final validation node.\n"
        "- Keep prompts concise and executable; use {N_output} placeholders when a node depends on prior output.\n"
        "- Preserve semantic roles in node.metadata.original_node_type when useful.\n"
        "- The output must match this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "ALFWorld compact sample:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def validate_llm_dag(dag: dict[str, Any]) -> None:
    for field in ("dag_id", "original_question", "nodes", "edges"):
        if field not in dag:
            raise ValueError(f"Converted DAG is missing {field}")
    if not isinstance(dag["nodes"], list) or len(dag["nodes"]) < 4:
        raise ValueError("Converted DAG must contain at least four nodes")
    if not isinstance(dag["edges"], list) or not dag["edges"]:
        raise ValueError("Converted DAG edges must be a non-empty list")

    node_ids: list[str] = []
    node_types: set[str] = set()
    for index, node in enumerate(dag["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be a JSON object")
        if "node_id" not in node or "prompt" not in node:
            raise ValueError("Each node must include node_id and prompt")
        node_id = str(node["node_id"])
        if node_id != f"N{index}":
            raise ValueError("Node ids must be contiguous and start at N0")
        node_type = str(node.get("node_type", "")).upper()
        if node_type not in {"CPU", "GPU"}:
            raise ValueError(f"Node type must be CPU or GPU, got {node.get('node_type')!r}")
        node["node_type"] = node_type
        node_ids.append(node_id)
        node_types.add(node_type)

    if len(node_types) < 2:
        raise ValueError("Converted DAG should contain both CPU and GPU nodes")

    node_id_set = set(node_ids)
    for index, edge in enumerate(dag["edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"Edge {index} must be a JSON object")
        if edge.get("source") not in node_id_set or edge.get("target") not in node_id_set:
            raise ValueError(f"Edge references unknown node: {edge}")

    dag.setdefault("topology", "mixed")
    dag.setdefault("question_type", "embodied_task")
    dag.setdefault("gold_answer", "")
    dag.setdefault("hop_count", max(len(dag["nodes"]) - 1, 1))
    dag.setdefault("metadata", {})
    if not isinstance(dag["metadata"], dict):
        raise ValueError("Converted DAG metadata must be a JSON object")
    dag["metadata"].setdefault("conversion", "llm")


def llm_http_status(exc: BaseException) -> int | None:
    match = re.search(r"\bHTTP Error (\d{3})\b", str(exc))
    return int(match.group(1)) if match else None


def is_non_retryable_llm_error(exc: BaseException) -> bool:
    return llm_http_status(exc) in NON_RETRYABLE_HTTP_CODES


def convert_one_llm(candidate: dict[str, Any], output_index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    path = Path(candidate["path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    annotation = choose_annotation(data)
    last_error: Exception | None = None

    for attempt in range(args.llm_retries + 1):
        try:
            if attempt:
                print(
                    f"Retrying ALFWorld source index {candidate['ordered_index']} "
                    f"attempt {attempt + 1}/{args.llm_retries + 1}"
                )
            dag = parse_json_object(call_llm(build_llm_prompt(candidate, data, annotation), args))
            validate_llm_dag(dag)
            dag["dag_id"] = f"alfworld_{output_index + 1:04d}"
            dag.setdefault("metadata", {})
            dag["metadata"].setdefault("source", "ALFWorld")
            dag["metadata"].setdefault("split", candidate["split"])
            dag["metadata"].setdefault("task_type", candidate["task_type"])
            dag["metadata"].setdefault("task_instance", candidate["task_instance"])
            dag["metadata"].setdefault("source_index", candidate["ordered_index"])
            dag["metadata"]["conversion"] = "llm"
            return dag
        except (
            RuntimeError,
            ValueError,
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if is_non_retryable_llm_error(exc):
                if getattr(args, "allow_rule_fallback", False):
                    print(
                        f"LLM conversion failed for source index {candidate['ordered_index']}; "
                        f"using rule fallback: {exc}"
                    )
                    return build_dag(candidate, output_index)
                raise FatalLLMError(
                    f"LLM provider rejected source index {candidate['ordered_index']}: {exc}"
                ) from exc
            if attempt < args.llm_retries:
                time.sleep(0.5 * (attempt + 1))

    if getattr(args, "allow_rule_fallback", False):
        print(
            f"LLM conversion failed for source index {candidate['ordered_index']}; "
            f"using rule fallback: {last_error}"
        )
        return build_dag(candidate, output_index)
    if args.skip_invalid_llm:
        print(f"Skipping source index {candidate['ordered_index']}; LLM conversion failed: {last_error}")
        return None
    raise RuntimeError(
        f"LLM conversion failed for source index {candidate['ordered_index']}: {last_error}"
    ) from last_error


def build_dag(candidate: dict[str, Any], output_index: int) -> dict[str, Any]:
    path = Path(candidate["path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    annotation = choose_annotation(data)

    task_type = clean_text(data.get("task_type", candidate["task_type"]))
    task_desc = clean_text(annotation.get("task_desc", "")) or f"ALFWorld task: {task_type}"
    task_desc_display = task_desc.rstrip(".")
    pddl_params = extract_pddl_params(data)
    scene_info = extract_scene_info(data)
    branches = branch_specs(pddl_params)
    actions = action_steps(data, annotation)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    def add_node(node_type: str, prompt: str, metadata: dict[str, Any] | None = None) -> str:
        node_id = f"N{len(nodes)}"
        node: dict[str, Any] = {
            "node_id": node_id,
            "node_type": node_type,
            "prompt": prompt,
        }
        if metadata:
            node["metadata"] = metadata
        nodes.append(node)
        return node_id

    plan_node = add_node(
        "GPU",
        (
            f"Interpret this ALFWorld task and extract the success criteria: {task_desc_display}. "
            f"Task type: {task_type}. Scene: {scene_info.get('floor_plan', '') or scene_info.get('scene_num', '')}."
        ),
        {
            "original_node_type": "task_planning",
            "task_type": task_type,
        },
    )

    branch_ids: list[str] = []
    for branch in branches:
        branch_id = add_node(
            "GPU",
            (
                f"{branch['label']} for the ALFWorld task '{task_desc_display}'. "
                f"Known value: {branch['value']}. Use this to plan the execution order."
            ),
            {
                "original_node_type": "constraint_extraction",
                "branch_key": branch["key"],
                "branch_value": branch["value"],
                "task_type": task_type,
            },
        )
        edges.append({"source": plan_node, "target": branch_id})
        branch_ids.append(branch_id)

    merge_inputs = branch_ids or [plan_node]
    merge_placeholders = " and ".join(f"{{{node_id}_output}}" for node_id in merge_inputs)
    merge_prompt = (
        f"Combine the planning outputs for '{task_desc_display}' into one executable ALFWorld plan. "
        f"Use the extracted constraints and the expert trajectory to decide the action order. "
        f"Inputs: {merge_placeholders}."
    )
    merge_node = add_node(
        "GPU",
        merge_prompt,
        {
            "original_node_type": "plan_merge",
            "task_type": task_type,
        },
    )
    for branch_id in branch_ids or [plan_node]:
        edges.append({"source": branch_id, "target": merge_node})

    previous = merge_node
    for step in actions:
        step_desc_display = step["description"].rstrip(".")
        action_prompt = (
            f"Execute this ALFWorld high-level step for '{task_desc_display}': {step_desc_display}. "
            f"Underlying action: {step['action_repr']}. "
            f"Use the previous step output {{{previous}_output}} to continue the episode."
        )
        action_node = add_node(
            "CPU",
            action_prompt,
            {
                "original_node_type": "high_level_action",
                "task_type": task_type,
                "high_idx": step["high_idx"],
                "description": step["description"],
                "action_repr": step["action_repr"],
            },
        )
        edges.append({"source": previous, "target": action_node})
        previous = action_node

    add_node(
        "GPU",
        (
            f"Verify that the ALFWorld task '{task_desc_display}' is complete and summarize "
            "the achieved state in one concise sentence."
        ),
        {
            "original_node_type": "validation",
            "task_type": task_type,
        },
    )
    final_node = nodes[-1]["node_id"]
    edges.append({"source": previous, "target": final_node})

    original_question = task_desc
    gold_answer = task_desc
    return {
        "dag_id": f"alfworld_{output_index + 1:04d}",
        "topology": "mixed",
        "question_type": "embodied_task",
        "original_question": original_question,
        "gold_answer": gold_answer,
        "nodes": nodes,
        "edges": edges,
        "hop_count": max(len(nodes) - 1, 1),
        "metadata": {
            "source": "ALFWorld",
            "split": candidate["split"],
            "task_type": task_type,
            "task_instance": candidate["task_instance"],
            "source_path": str(path.as_posix()),
            "source_index": candidate["ordered_index"],
            "scene_num": scene_info.get("scene_num"),
            "floor_plan": scene_info.get("floor_plan"),
            "pddl_params": pddl_params,
            "high_level_actions": len(actions),
            "branches": [branch["key"] for branch in branches],
            "conversion": "rule_template",
        },
    }


def validate_dag_shape(dag: dict[str, Any]) -> None:
    for field in ("dag_id", "original_question", "nodes", "edges"):
        if field not in dag:
            raise ValueError(f"Converted DAG is missing {field}")
    if not isinstance(dag["nodes"], list) or not dag["nodes"]:
        raise ValueError("Converted DAG must contain nodes")
    if not isinstance(dag["edges"], list):
        raise ValueError("Converted DAG edges must be a list")

    node_ids: list[str] = []
    for node in dag["nodes"]:
        if "node_id" not in node or "prompt" not in node:
            raise ValueError("Each node must include node_id and prompt")
        node_type = str(node.get("node_type", "")).upper()
        if node_type not in {"CPU", "GPU"}:
            raise ValueError(f"Node type must be CPU or GPU, got {node.get('node_type')!r}")
        node["node_type"] = node_type
        node_ids.append(str(node["node_id"]))

    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids):
        raise ValueError("Node ids must be unique")
    for edge in dag["edges"]:
        if edge.get("source") not in node_id_set or edge.get("target") not in node_id_set:
            raise ValueError(f"Edge references unknown node: {edge}")

    dag.setdefault("topology", "mixed")
    dag.setdefault("question_type", "embodied_task")
    dag.setdefault("gold_answer", "")
    dag.setdefault("hop_count", max(len(dag["nodes"]) - 1, 1))
    dag.setdefault("metadata", {})
    dag["metadata"].setdefault("conversion", "rule_template")


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_output(output_path: Path, dags: list[dict[str, Any]]) -> None:
    dags = standardize_dags(dags)
    write_json_atomic(output_path, dags)
    write_json_atomic(resource_estimate_path(output_path), build_resource_estimates(dags))


def load_existing_output(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    data = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Existing output is not a JSON list: {output_path}")
    return data


def get_source_index(dag: dict[str, Any]) -> int | None:
    metadata = dag.get("metadata") if isinstance(dag.get("metadata"), dict) else {}
    source_index = metadata.get("source_index")
    if isinstance(source_index, bool):
        return None
    try:
        return int(source_index)
    except (TypeError, ValueError):
        return None


def pending_candidates(
    ordered: list[dict[str, Any]],
    dags: list[dict[str, Any]],
    start: int,
    limit: int,
) -> list[dict[str, Any]]:
    existing_indexes = [index for dag in dags if (index := get_source_index(dag)) is not None]
    if len(existing_indexes) != len(set(existing_indexes)):
        raise ValueError("Existing output contains duplicate metadata.source_index values")

    completed = set(existing_indexes)
    pending = [
        candidate
        for candidate in ordered
        if candidate["ordered_index"] >= start and candidate["ordered_index"] not in completed
    ]
    if limit > 0:
        pending = pending[: max(limit - len(dags), 0)]
    return pending


def checkpoint_before_abort(output_path: Path, dags: list[dict[str, Any]]) -> None:
    if dags:
        write_output(output_path, dags)
        print(f"Preserved {len(dags)} completed DAGs in {output_path}")


def enforce_failure_limit(
    args: argparse.Namespace,
    consecutive_failures: int,
    output_path: Path,
    dags: list[dict[str, Any]],
) -> None:
    if args.max_consecutive_failures <= 0 or consecutive_failures < args.max_consecutive_failures:
        return
    checkpoint_before_abort(output_path, dags)
    raise RuntimeError(
        f"Aborting after {consecutive_failures} consecutive LLM conversion failures. "
        "Check provider availability, quota, model name, and logs before resuming."
    )


def convert(args: argparse.Namespace, output_path: Path) -> list[dict[str, Any]]:
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer")
    if args.start < 0 or args.limit < 0:
        raise ValueError("--start and --limit must be non-negative")
    if args.llm_retries < 0 or args.checkpoint_every < 0 or args.max_consecutive_failures < 0:
        raise ValueError("Retry, checkpoint, and failure-limit arguments must be non-negative")
    if args.llm_timeout <= 0:
        raise ValueError("--llm-timeout must be greater than zero")

    input_root = Path(args.input_root)
    candidates = collect_candidates(input_root, args.splits)
    if not candidates:
        raise ValueError(f"No ALFWorld traj_data.json files found under {input_root}")

    ordered = order_candidates(candidates, args.sample_mode)

    dags = load_existing_output(output_path)
    if dags:
        print(f"Resuming from existing output with {len(dags)} DAGs: {output_path}")
    pending = pending_candidates(ordered, dags, args.start, args.limit)
    if not pending:
        return dags

    if args.converter == "llm" and args.workers > 1:
        return convert_parallel_llm(args, output_path, pending, dags)

    consecutive_failures = 0
    target_count = args.limit if args.limit > 0 else len(dags) + len(pending)
    for candidate in pending:
        print(
            f"Converting ALFWorld sample {len(dags) + 1}/"
            f"{target_count} "
            f"(split={candidate['split']}, task={candidate['task_type']})"
        )
        try:
            if args.converter == "llm":
                dag = convert_one_llm(candidate, len(dags), args)
            else:
                dag = build_dag(candidate, len(dags))
        except FatalLLMError:
            checkpoint_before_abort(output_path, dags)
            raise
        if dag is None:
            consecutive_failures += 1
            enforce_failure_limit(args, consecutive_failures, output_path, dags)
            continue
        consecutive_failures = 0
        validate_dag_shape(dag)
        dags.append(dag)
        if args.checkpoint_every > 0 and len(dags) % args.checkpoint_every == 0:
            write_output(output_path, dags)
            print(f"Checkpoint wrote {len(dags)} DAGs to {output_path}")

    return dags


def convert_parallel_llm(
    args: argparse.Namespace,
    output_path: Path,
    pending: list[dict[str, Any]],
    dags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert independent candidates concurrently while preserving output order.

    ALFWorld candidates are independent at conversion time. A bounded thread pool
    overlaps API latency without allowing an unbounded request fan-out. Results
    are appended in candidate order, and DAG ids are reassigned after each batch
    so skipped invalid responses cannot create duplicate ids.
    """
    cursor = 0
    last_checkpoint_count = len(dags)
    consecutive_failures = 0
    while cursor < len(pending):
        batch_size = min(args.workers, len(pending) - cursor)
        batch = pending[cursor : cursor + batch_size]
        for candidate in batch:
            print(
                f"Converting ALFWorld candidate {candidate['ordered_index']} "
                f"(split={candidate['split']}, task={candidate['task_type']})"
            )

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(convert_one_llm, candidate, len(dags) + offset, args)
                for offset, candidate in enumerate(batch)
            ]
            results: list[dict[str, Any] | None] = []
            fatal_error: FatalLLMError | None = None
            for future in futures:
                try:
                    results.append(future.result())
                except FatalLLMError as exc:
                    results.append(None)
                    fatal_error = fatal_error or exc

        for dag in results:
            if dag is None:
                consecutive_failures += 1
                if fatal_error is None:
                    enforce_failure_limit(args, consecutive_failures, output_path, dags)
                continue
            consecutive_failures = 0
            dag["dag_id"] = f"alfworld_{len(dags) + 1:04d}"
            validate_dag_shape(dag)
            dags.append(dag)

        cursor += batch_size
        if fatal_error is not None:
            checkpoint_before_abort(output_path, dags)
            raise fatal_error
        if (
            args.checkpoint_every > 0
            and len(dags) >= last_checkpoint_count + args.checkpoint_every
        ):
            write_output(output_path, dags)
            last_checkpoint_count = len(dags)
            print(f"Checkpoint wrote {len(dags)} DAGs to {output_path}")

    return dags
