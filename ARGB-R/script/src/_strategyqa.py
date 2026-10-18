"""Convert StrategyQA JSONL examples into autonomous CPU/GPU ARGB DAG JSON."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import http.client
import json
import time
import urllib.error
from pathlib import Path
from typing import Any

from .dag_conversion_common import call_llm, parse_json_object, validate_dag_shape
from .dag_standard import build_resource_estimates, resource_estimate_path, standardize_dags


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def convert_item(item: dict[str, Any], output_index: int) -> dict[str, Any]:
    question = str(item.get("question", ""))
    decomposition = item.get("decomposition") or []
    nodes = [
        {
            "node_id": "N0",
            "node_type": "GPU",
            "prompt": f"Plan the StrategyQA yes/no reasoning steps for this question: {question}",
            "metadata": {
                "operation_type": "query_planning",
                "device_reason": "Planning a multi-hop natural-language reasoning path benefits from model inference.",
            },
        }
    ]
    edges = []
    previous = "N0"

    for step_index, subquestion in enumerate(decomposition, start=1):
        node_id = f"N{step_index}"
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "GPU",
                "prompt": (
                    f"Answer this StrategyQA subquestion using the provided facts: {subquestion}\n"
                    f"Original question: {question}\n"
                    f"Previous result: {{{previous}_output}}"
                ),
                "metadata": {
                    "operation_type": "reasoning",
                    "device_reason": "The node performs evidence-grounded natural-language reasoning.",
                },
            }
        )
        edges.append({"source": previous, "target": node_id})
        previous = node_id

    final_node_id = f"N{len(nodes)}"
    nodes.append(
        {
            "node_id": final_node_id,
            "node_type": "CPU",
            "prompt": f"Format {{{previous}_output}} as a boolean yes/no answer for: {question}",
            "metadata": {
                "operation_type": "formatting",
                "device_reason": "Formatting a known boolean answer is lightweight deterministic work.",
            },
        }
    )
    edges.append({"source": previous, "target": final_node_id})

    return {
        "dag_id": f"strategyqa_{output_index + 1:04d}",
        "topology": "linear",
        "question_type": "yes_no_multi_hop_qa",
        "original_question": question,
        "gold_answer": str(item.get("answer", "")),
        "nodes": nodes,
        "edges": edges,
        "hop_count": max(len(nodes) - 1, 1),
        "metadata": {
            "source": "StrategyQA",
            "source_id": item.get("qid", ""),
            "source_index": output_index,
            "term": item.get("term", ""),
            "conversion": "decomposition_rule",
        },
    }


def build_llm_prompt(item: dict[str, Any], output_index: int) -> str:
    question = str(item.get("question", ""))
    answer = item.get("answer")
    compact = {
        "source_index": output_index,
        "source_id": item.get("qid", ""),
        "term": item.get("term", ""),
        "description": item.get("description", ""),
        "question": question,
        "answer": answer,
        "facts": item.get("facts", []),
    }
    schema = {
        "dag_id": f"strategyqa_{output_index + 1:04d}",
        "topology": "linear|parallel|diamond|mixed",
        "question_type": "yes_no_multi_hop_qa",
        "original_question": question,
        "gold_answer": str(answer),
        "nodes": [
            {
                "node_id": "N0",
                "node_type": "CPU|GPU",
                "prompt": "clear executable prompt for this schedulable node",
                "metadata": {
                    "operation_type": "query_planning|evidence_retrieval|entity_resolution|reasoning|comparison|aggregation|validation|formatting",
                    "device_reason": "why this node is better scheduled on CPU or GPU",
                },
            }
        ],
        "edges": [{"source": "N0", "target": "N1"}],
        "hop_count": 1,
        "metadata": {
            "source": "StrategyQA",
            "source_id": item.get("qid", ""),
            "source_index": output_index,
            "conversion": "llm_autonomous",
        },
    }
    return (
        "You autonomously convert one StrategyQA yes/no reasoning example into a compact ARGB DAG JSON.\n"
        "Wenbo uses DAGs as schedulable execution graphs: each node is an independent unit of work, and edges "
        "represent data or control dependencies. The runtime schedules independent nodes concurrently across CPU "
        "and GPU resources to reduce the critical path and improve throughput while preserving the reasoning "
        "needed to answer the question.\n"
        "Infer the reasoning plan directly from the question, term description, and facts. Do not rely on a "
        "provided decomposition. Choose the topology, nodes, edges, and CPU/GPU placement that best represent "
        "the true evidence and reasoning dependencies.\n"
        "Rules:\n"
        "- Return exactly one valid JSON object and no Markdown.\n"
        "- Preserve original_question and gold_answer.\n"
        "- Use node ids N0, N1, N2, ... with no gaps.\n"
        "- node_type must be either CPU or GPU.\n"
        "- For StrategyQA, natural-language interpretation, evidence selection from facts, entity resolution, "
        "implicit bridge reasoning, comparison reasoning, and yes/no answer generation should normally be GPU nodes.\n"
        "- Use CPU nodes only for lightweight deterministic work such as routing, exact boolean normalization, "
        "string/date/number comparison when all values are already extracted, final formatting, validation, and "
        "other low-latency control logic.\n"
        "- Each answerable DAG should include at least one GPU node; avoid all-CPU DAGs.\n"
        "- Put the semantic role of each node in metadata.operation_type.\n"
        "- Put a short scheduling rationale in metadata.device_reason explaining why CPU or GPU is appropriate.\n"
        "- Expose parallelism whenever possible: independent fact checks should be sibling nodes, not a forced chain.\n"
        "- Prefer a scheduling-oriented DAG over a human step-by-step explanation: create sibling GPU nodes for "
        "independent fact checks, entity/background lookups, and competing yes/no evidence paths.\n"
        "- For comparison questions, extract or reason about each side in parallel, then use a CPU comparison node "
        "only if both compared values are already explicit and deterministic.\n"
        "- For causal, commonsense, or implicit bridge questions, keep the neural reasoning step on GPU and separate "
        "any deterministic boolean normalization into a CPU node.\n"
        "- Aim for 4-10 nodes for typical answerable examples when the facts support parallel work; fewer nodes are "
        "acceptable only when the question has a single true dependency chain.\n"
        "- Use linear dependencies only when one node truly needs another node's output.\n"
        "- Keep the DAG compact; split work only when it creates a meaningful scheduling boundary or captures a "
        "real reasoning dependency.\n"
        "- Keep prompts concise and self-contained; use {N_output} placeholders when a node needs prior output.\n"
        "- Include edge data_volume_bytes when you can estimate the amount of intermediate data passed between nodes.\n"
        "- Set metadata.conversion to llm_autonomous.\n"
        "- The output must match this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "StrategyQA compact sample:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def convert_item_with_llm(item: dict[str, Any], output_index: int, args: argparse.Namespace) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.llm_retries + 1):
        try:
            if attempt:
                print(f"Retrying source index {output_index}, attempt {attempt + 1}/{args.llm_retries + 1}")
            dag = parse_json_object(call_llm(build_llm_prompt(item, output_index), args))
            validate_dag_shape(dag)
            dag["dag_id"] = f"strategyqa_{output_index + 1:04d}"
            dag.setdefault("metadata", {})
            dag["metadata"].setdefault("source", "StrategyQA")
            dag["metadata"].setdefault("source_id", item.get("qid", ""))
            dag["metadata"].setdefault("source_index", output_index)
            dag["metadata"].setdefault("term", item.get("term", ""))
            dag["metadata"]["conversion"] = "llm_autonomous"
            return dag
        except (
            RuntimeError,
            ValueError,
            TimeoutError,
            OSError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if not args.allow_rule_fallback:
        raise RuntimeError(f"LLM conversion failed for source index {output_index}: {last_error}") from last_error
    print(f"LLM conversion failed for source index {output_index}; using rule fallback: {last_error}")
    return convert_item(item, output_index)


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
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Existing output is not a JSON list: {output_path}")
    return data


def get_source_index(dag: dict[str, Any]) -> int | None:
    metadata = dag.get("metadata") if isinstance(dag.get("metadata"), dict) else {}
    raw_index = metadata.get("source_index")
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return None


def sorted_dags(dags_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dags_by_index[index] for index in sorted(dags_by_index)]


def convert_single(item: dict[str, Any], item_index: int, args: argparse.Namespace, use_llm: bool) -> dict[str, Any]:
    return convert_item_with_llm(item, item_index, args) if use_llm else convert_item(item, item_index)


def convert(args: argparse.Namespace, output_path: Path) -> list[dict[str, Any]]:
    all_items = iter_jsonl(Path(args.input))
    end_index = min(len(all_items), args.start + args.limit)
    target_indices = set(range(args.start, end_index))
    use_llm = args.converter == "llm" or args.use_llm

    dags_by_index: dict[int, dict[str, Any]] = {}
    for dag in load_existing_output(output_path):
        source_index = get_source_index(dag)
        if source_index in target_indices:
            dags_by_index[source_index] = dag

    if dags_by_index:
        print(f"Resuming from existing output with {len(dags_by_index)} completed DAGs: {output_path}")

    pending = [(index, all_items[index]) for index in range(args.start, end_index) if index not in dags_by_index]
    if not pending:
        return sorted_dags(dags_by_index)

    checkpoint_every = max(args.checkpoint_every, 0)
    completed_since_checkpoint = 0
    total_target = len(target_indices)

    if not use_llm or args.workers <= 1:
        for item_index, item in pending:
            print(f"Converting source index {item_index} ({len(dags_by_index) + 1}/{total_target})")
            dags_by_index[item_index] = convert_single(item, item_index, args, use_llm)
            completed_since_checkpoint += 1
            if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
                write_output(output_path, sorted_dags(dags_by_index))
                completed_since_checkpoint = 0
                print(f"Checkpoint wrote {len(dags_by_index)} DAGs to {output_path}")
        return sorted_dags(dags_by_index)

    max_workers = max(1, args.workers)
    print(f"Using {max_workers} workers for {len(pending)} pending StrategyQA conversions")
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index: dict[Future[dict[str, Any]], int] = {
            executor.submit(convert_single, item, item_index, args, use_llm): item_index
            for item_index, item in pending
        }
        try:
            for future in as_completed(future_to_index):
                item_index = future_to_index[future]
                try:
                    dags_by_index[item_index] = future.result()
                except Exception as exc:
                    failures[item_index] = str(exc)
                    print(f"Failed source index {item_index}: {exc}")
                else:
                    completed_since_checkpoint += 1
                    print(f"Converted source index {item_index} ({len(dags_by_index)}/{total_target})")
                if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
                    write_output(output_path, sorted_dags(dags_by_index))
                    completed_since_checkpoint = 0
                    print(f"Checkpoint wrote {len(dags_by_index)} DAGs to {output_path}")
        except BaseException:
            write_output(output_path, sorted_dags(dags_by_index))
            for future in future_to_index:
                future.cancel()
            raise

    if failures:
        write_output(output_path, sorted_dags(dags_by_index))
        failed_indices = ", ".join(str(index) for index in sorted(failures)[:20])
        extra = "" if len(failures) <= 20 else f", ... +{len(failures) - 20} more"
        raise RuntimeError(
            f"{len(failures)} StrategyQA conversions failed; checkpoint saved to {output_path}. "
            f"Failed source indices: {failed_indices}{extra}"
        )

    return sorted_dags(dags_by_index)
