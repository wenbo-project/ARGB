"""Internal ToolBench backend for the unified ARGB-R converter.

The converter intentionally keeps the output compact: ToolBench function
observations can contain large binary-like payloads, so DAG nodes keep only the
task text and assistant action/action-input pairs.

Two conversion modes are supported:
1. rule-based linear conversion, which is deterministic and offline;
2. LLM conversion, which asks an OpenAI-compatible endpoint to infer nodes and
   dependencies from the compact ToolBench trajectory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import http.client
import json
import re
import time
import urllib.error
from pathlib import Path
from typing import Any

from .dag_conversion_common import call_llm, parse_json_object
from .dag_standard import build_resource_estimates, resource_estimate_path, standardize_dags


ACTION_RE = re.compile(
    r"Action\s*:\s*(?P<action>[^\n]+)\n\s*Action Input\s*:\s*(?P<input>.*)",
    re.IGNORECASE | re.DOTALL,
)
TOOL_NAME_RE = re.compile(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]")




def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_task(conversations: list[dict[str, Any]], fallback: str) -> str:
    for message in conversations:
        if message.get("from") == "user":
            value = str(message.get("value", "")).strip()
            value = value.replace("Begin!", "").strip()
            if value:
                return clean_text(value)
    return clean_text(fallback)


def extract_tool_names(conversations: list[dict[str, Any]]) -> list[str]:
    for message in conversations:
        if message.get("from") == "system":
            names = TOOL_NAME_RE.findall(str(message.get("value", "")))
            return sorted(dict.fromkeys(names))
    return []


def extract_actions(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    actions = []
    for message in conversations:
        if message.get("from") != "assistant":
            continue
        value = str(message.get("value", "")).strip()
        match = ACTION_RE.search(value)
        if not match:
            continue
        action = match.group("action").strip()
        action_input = match.group("input").strip()
        actions.append(
            {
                "action": action,
                "action_input": action_input,
            }
        )
    return actions


def truncate_text(value: str, max_chars: int) -> str:
    value = clean_text(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def extract_steps(conversations: list[dict[str, Any]], observation_chars: int) -> list[dict[str, str]]:
    steps = []
    pending_action: dict[str, str] | None = None
    for message in conversations:
        role = message.get("from")
        value = str(message.get("value", "")).strip()
        if role == "assistant":
            match = ACTION_RE.search(value)
            if not match:
                continue
            pending_action = {
                "action": match.group("action").strip(),
                "action_input": match.group("input").strip(),
                "observation": "",
            }
            steps.append(pending_action)
        elif role == "function" and pending_action is not None:
            pending_action["observation"] = truncate_text(value, observation_chars)
            pending_action = None
    return steps


def make_node_prompt(task: str, action: dict[str, str], previous_node_id: str) -> str:
    return (
        f"Original ToolBench task: {task}\n"
        f"Previous step output: {{{previous_node_id}_output}}\n"
        f"Call action `{action['action']}` with this action input:\n"
        f"{action['action_input']}"
    )


def convert_item(item: dict[str, Any], item_index: int) -> dict[str, Any] | None:
    conversations = item.get("conversations", [])
    if not isinstance(conversations, list):
        return None

    task = extract_task(conversations, fallback=str(item.get("id", "")))
    tool_names = extract_tool_names(conversations)
    actions = extract_actions(conversations)
    if not actions:
        return None

    dag_id = f"toolbench_{item_index + 1:04d}"
    nodes = [
        {
            "node_id": "N0",
            "node_type": "llm_infer",
            "prompt": (
                "Understand the ToolBench task and identify the available tools.\n"
                f"Task: {task}\n"
                f"Available action names: {', '.join(tool_names) if tool_names else 'unknown'}"
            ),
        }
    ]
    edges = []
    previous_node_id = "N0"

    for action_index, action in enumerate(actions, start=1):
        node_id = f"N{action_index}"
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "tool_call",
                "prompt": make_node_prompt(task, action, previous_node_id),
                "metadata": {
                    "action": action["action"],
                    "action_input": action["action_input"],
                },
            }
        )
        edges.append({"source": previous_node_id, "target": node_id})
        previous_node_id = node_id

    final_node_id = f"N{len(nodes)}"
    nodes.append(
        {
            "node_id": final_node_id,
            "node_type": "aggregate",
            "prompt": (
                f"Use the result from {{{previous_node_id}_output}} to produce the final answer "
                f"for the original ToolBench task: {task}"
            ),
        }
    )
    edges.append({"source": previous_node_id, "target": final_node_id})

    return {
        "dag_id": dag_id,
        "topology": "linear",
        "question_type": "tool_use",
        "original_question": task,
        "gold_answer": "",
        "nodes": nodes,
        "edges": edges,
        "hop_count": max(len(nodes) - 1, 1),
        "metadata": {
            "source": "ToolBench",
            "source_id": item.get("id", ""),
            "source_index": item_index,
            "tool_names": tool_names,
            "action_count": len(actions),
        },
    }


def build_llm_prompt(item: dict[str, Any], item_index: int, observation_chars: int) -> str | None:
    conversations = item.get("conversations", [])
    if not isinstance(conversations, list):
        return None

    task = extract_task(conversations, fallback=str(item.get("id", "")))
    tool_names = extract_tool_names(conversations)
    steps = extract_steps(conversations, observation_chars=observation_chars)
    if not steps:
        return None

    compact = {
        "source_index": item_index,
        "source_id": item.get("id", ""),
        "task": task,
        "available_tools": tool_names,
        "trajectory_steps": steps,
    }
    schema = {
        "dag_id": f"toolbench_{item_index + 1:04d}",
        "topology": "linear|parallel|diamond|mixed",
        "question_type": "tool_use",
        "original_question": task,
        "gold_answer": "",
        "nodes": [
            {
                "node_id": "N0",
                "node_type": "CPU|GPU",
                "prompt": "clear executable prompt for this node",
            }
        ],
        "edges": [{"source": "N0", "target": "N1"}],
        "hop_count": 1,
        "metadata": {
            "source": "ToolBench",
            "source_id": item.get("id", ""),
            "source_index": item_index,
            "conversion": "llm",
        },
    }
    return (
        "You convert ToolBench agent trajectories into compact ARGB DAG JSON.\n"
        "Infer the minimal useful DAG structure from the task, available tools, actions, and observations.\n"
        "Rules:\n"
        "- Return exactly one JSON object and no Markdown.\n"
        "- Preserve the original user task in original_question.\n"
        "- Use stable node ids N0, N1, N2, ... with no gaps.\n"
        "- Node types must be exactly CPU or GPU.\n"
        "- Use CPU for concrete tool/API/browser/search/file actions and other external work.\n"
        "- Use GPU for planning, parameter derivation, reasoning, retrieval-as-generation, and final synthesis.\n"
        "- Add edges only for real control/data dependencies.\n"
        "- Do not make a purely sequential chain unless every step truly depends on the previous step.\n"
        "- If two tool calls or reasoning branches are independent, they must share the same parent and run in parallel.\n"
        "- Prefer a mixed DAG with fan-out/fan-in when the trajectory has independent tools, independent parameters, or independent subgoals.\n"
        "- Keep prompts concise but self-contained; use {N_output} placeholders when a node needs prior output.\n"
        "- Do not include raw function observations that look like binary data.\n"
        "- Preserve semantic roles, if useful, in node.metadata.original_node_type such as llm_infer, tool_call, retrieval, or aggregate.\n"
        "- The output must match this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "ToolBench compact trajectory:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def validate_dag_shape(dag: dict[str, Any], *, require_nonlinear: bool = False) -> None:
    required = {"dag_id", "original_question", "nodes", "edges"}
    missing = required - set(dag)
    if missing:
        raise ValueError(f"Converted DAG is missing fields: {sorted(missing)}")
    nodes = dag["nodes"]
    edges = dag["edges"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Converted DAG must contain a non-empty nodes list")
    if not isinstance(edges, list):
        raise ValueError("Converted DAG edges must be a list")

    node_ids = []
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Each node must be an object")
        if "node_id" not in node or "prompt" not in node:
            raise ValueError("Each node must include node_id and prompt")
        node_type = str(node.get("node_type", "")).upper()
        if node_type not in {"CPU", "GPU"}:
            raise ValueError(f"Node type must be CPU or GPU, got {node.get('node_type')!r}")
        node["node_type"] = node_type
        node_ids.append(node["node_id"])
    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids):
        raise ValueError("Node ids must be unique")

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Each edge must be an object")
        if edge.get("source") not in node_id_set or edge.get("target") not in node_id_set:
            raise ValueError(f"Edge references unknown node: {edge}")

    if require_nonlinear and not _has_fanout_or_fanin(nodes, edges):
        raise ValueError("Converted DAG is not useful for parallel scheduling; expected at least one real fan-out or fan-in")

    dag.setdefault("topology", "mixed")
    dag.setdefault("question_type", "tool_use")
    dag.setdefault("gold_answer", "")
    dag.setdefault("hop_count", max(len(nodes) - 1, 1))
    dag.setdefault("metadata", {})
    dag["metadata"].setdefault("conversion", "llm")


def _has_fanout_or_fanin(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bool:
    if len(nodes) < 3:
        return False
    in_degree = {node["node_id"]: 0 for node in nodes}
    out_degree = {node["node_id"]: 0 for node in nodes}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in out_degree:
            out_degree[source] += 1
        if target in in_degree:
            in_degree[target] += 1
    return max(in_degree.values(), default=0) > 1 or max(out_degree.values(), default=0) > 1


def convert_item_with_llm(item: dict[str, Any], item_index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    prompt = build_llm_prompt(item, item_index, observation_chars=args.observation_chars)
    if prompt is None:
        return None
    output = call_llm(
        prompt,
        base_url=args.llm_base_url,
        llm_api=args.llm_api,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )
    dag = parse_json_object(output)
    validate_dag_shape(dag, require_nonlinear=args.require_nonlinear)
    dag["dag_id"] = f"toolbench_{item_index + 1:04d}"
    metadata = dag.setdefault("metadata", {})
    metadata.setdefault("source", "ToolBench")
    metadata.setdefault("source_id", item.get("id", ""))
    metadata.setdefault("source_index", item_index)
    metadata["conversion"] = "llm"
    return dag


def convert_with_retries(item: dict[str, Any], item_index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    last_error: Exception | None = None
    for attempt in range(args.llm_retries + 1):
        try:
            if attempt:
                print(f"Retrying source index {item_index}, attempt {attempt + 1}/{args.llm_retries + 1}")
            return convert_item_with_llm(item, item_index, args)
        except (RuntimeError, ValueError, urllib.error.URLError, http.client.IncompleteRead, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if args.no_rule_fallback:
        if args.skip_invalid_llm:
            print(f"Skipping source index {item_index}; LLM conversion did not satisfy constraints: {last_error}")
            return None
        raise RuntimeError(f"LLM conversion failed for source index {item_index}: {last_error}") from last_error
    print(f"LLM conversion failed for source index {item_index}; using rule fallback: {last_error}")
    return convert_item(item, item_index)


def write_checkpoint(dags: list[dict[str, Any]], output_path: Path) -> None:
    checkpoint_dags = standardize_dags(dags)
    output_path.write_text(json.dumps(checkpoint_dags, ensure_ascii=False, indent=2), encoding="utf-8")
    resource_estimate_path(output_path).write_text(
        json.dumps(build_resource_estimates(checkpoint_dags), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Checkpoint wrote {len(dags)} DAGs to {output_path}")


def convert_parallel(raw_items: list[dict[str, Any]], output_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    dags: list[dict[str, Any]] = []
    results: dict[int, dict[str, Any] | None] = {}
    pending: dict[Future[dict[str, Any] | None], int] = {}
    next_submit = args.start
    next_emit = args.start
    last_checkpoint = 0
    max_workers = max(args.workers, 1)
    max_pending = args.max_pending if args.max_pending > 0 else max_workers

    def submit_until_capacity(executor: ThreadPoolExecutor) -> None:
        nonlocal next_submit
        while next_submit < len(raw_items) and len(pending) < max_pending and len(dags) < args.limit:
            print(f"Scheduling source index {next_submit} ({len(dags) + len(pending) + 1}/{args.limit})")
            future = executor.submit(convert_with_retries, raw_items[next_submit], next_submit, args)
            pending[future] = next_submit
            next_submit += 1

    def maybe_checkpoint() -> None:
        nonlocal last_checkpoint
        if args.checkpoint_every > 0 and len(dags) >= last_checkpoint + args.checkpoint_every:
            write_checkpoint(dags, output_path)
            last_checkpoint = len(dags)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        submit_until_capacity(executor)
        while len(dags) < args.limit and (pending or results or next_submit < len(raw_items)):
            progressed = False
            while next_emit in results and len(dags) < args.limit:
                dag = results.pop(next_emit)
                next_emit += 1
                progressed = True
                if dag is None:
                    submit_until_capacity(executor)
                    continue
                dags.append(dag)
                maybe_checkpoint()
                submit_until_capacity(executor)

            if len(dags) >= args.limit:
                break
            if progressed:
                continue
            submit_until_capacity(executor)
            if not pending:
                break

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item_index = pending.pop(future)
                try:
                    results[item_index] = future.result()
                except Exception as exc:
                    if not args.skip_invalid_llm:
                        raise
                    print(f"Skipping source index {item_index}; worker failed: {exc}")
                    results[item_index] = None

        for future in pending:
            future.cancel()

    return dags


def convert(input_path: Path, output_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)
    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    if args.limit == 0:
        return []

    if args.use_llm and args.workers > 1:
        print(f"Running LLM conversion with {args.workers} workers")
        return convert_parallel(raw_items, output_path, args)

    dags = []
    for item_index, item in enumerate(raw_items[args.start :], start=args.start):
        if args.use_llm:
            print(f"Converting source index {item_index} ({len(dags) + 1}/{args.limit})")
            dag = convert_with_retries(item, item_index, args)
            if dag is not None:
                time.sleep(0.05)
        else:
            dag = convert_item(item, item_index)
        if dag is None:
            continue
        dags.append(dag)
        if args.use_llm and args.checkpoint_every > 0 and len(dags) % args.checkpoint_every == 0:
            write_checkpoint(dags, output_path)
        if len(dags) >= args.limit:
            break
    return dags
