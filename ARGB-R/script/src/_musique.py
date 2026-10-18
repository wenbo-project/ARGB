"""Convert MuSiQue JSONL examples into ARGB DAG JSON."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import http.client
import json
import re
import time
import urllib.error
from pathlib import Path
from typing import Any

from .dag_conversion_common import call_llm, parse_json_object
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
    decomposition = item.get("question_decomposition") or []
    nodes = [
        {
            "node_id": "N0",
            "node_type": "llm_infer",
            "prompt": f"Decompose the MuSiQue multi-hop question into ordered subquestions: {question}",
        }
    ]
    edges = []
    previous = "N0"

    for step_index, step in enumerate(decomposition, start=1):
        subquestion = str(step.get("question", ""))
        node_id = f"N{step_index}"
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "retrieval",
                "prompt": (
                    f"Answer this MuSiQue subquestion using the available paragraphs: {subquestion}\n"
                    f"Original question: {question}\n"
                    f"Previous result: {{{previous}_output}}"
                ),
                "metadata": {
                    "source_decomposition_id": step.get("id"),
                    "paragraph_support_idx": step.get("paragraph_support_idx"),
                },
            }
        )
        edges.append({"source": previous, "target": node_id})
        previous = node_id

    final_node_id = f"N{len(nodes)}"
    nodes.append(
        {
            "node_id": final_node_id,
            "node_type": "aggregate",
            "prompt": f"Use {{{previous}_output}} to answer the original MuSiQue question concisely: {question}",
        }
    )
    edges.append({"source": previous, "target": final_node_id})

    return {
        "dag_id": f"musique_{output_index + 1:04d}",
        "topology": "linear",
        "question_type": "multi_hop_qa",
        "original_question": question,
        "gold_answer": str(item.get("answer", "")),
        "nodes": nodes,
        "edges": edges,
        "hop_count": max(len(nodes) - 1, 1),
        "metadata": {
            "source": "MuSiQue",
            "source_id": item.get("id", ""),
            "source_index": output_index,
            "answerable": item.get("answerable"),
            "answer_aliases": item.get("answer_aliases", []),
            "conversion": "question_decomposition",
        },
    }


def build_llm_prompt(item: dict[str, Any], output_index: int, paragraph_max_chars: int = 900) -> str:
    question = str(item.get("question", ""))
    paragraphs = compact_paragraphs(item.get("paragraphs") or [], max_chars=paragraph_max_chars)
    compact = {
        "source_index": output_index,
        "source_id": item.get("id", ""),
        "question": question,
        "answer": str(item.get("answer", "")),
        "answerable": item.get("answerable"),
        "paragraphs": paragraphs,
    }
    schema = {
        "dag_id": f"musique_{output_index + 1:04d}",
        "topology": "linear|parallel|diamond|mixed",
        "question_type": "multi_hop_qa",
        "original_question": question,
        "gold_answer": str(item.get("answer", "")),
        "nodes": [
            {
                "node_id": "N0",
                "node_type": "CPU|GPU",
                "prompt": "executable node prompt",
                "metadata": {
                    "operation_type": "query_planning|evidence_retrieval|entity_resolution|reasoning|aggregation|validation|formatting",
                    "device_reason": "CPU/GPU scheduling rationale",
                },
            }
        ],
        "edges": [{"source": "N0", "target": "N1"}],
        "hop_count": 1,
        "metadata": {
            "source": "MuSiQue",
            "source_id": item.get("id", ""),
            "source_index": output_index,
            "conversion": "llm_autonomous",
        },
    }
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    sample_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return (
        "Convert this MuSiQue multi-hop QA sample into compact ARGB DAG JSON.\n"
        "Wenbo nodes are schedulable CPU/GPU work units; edges are true data/control dependencies.\n"
        "Use only the provided question and paragraphs. Preserve original_question, gold_answer, and source metadata.\n"
        "Output exactly one valid JSON object, no Markdown.\n"
        "Rules: node ids N0,N1,... no gaps; node_type is CPU or GPU. "
        "GPU handles semantic paragraph retrieval, matching, entity resolution, reasoning, and answer synthesis. "
        "CPU handles deterministic planning/routing, exact id/title selection, validation, normalization, and formatting "
        "after needed facts are explicit. Use only the nodes and edges needed for the real reasoning; expose useful "
        "parallel sibling work without inventing fake parallelism or fake chains. Node prompts must be executable, "
        "mention paragraph idx/title when useful, and use {N_output} when dependent on prior output. Fill "
        "metadata.operation_type, metadata.device_reason, and metadata.conversion='llm_autonomous'. Include "
        "edge data_volume_bytes only when reasonably estimable.\n"
        f"Schema:{schema_text}\n"
        f"Sample:{sample_text}"
    )


def compact_paragraphs(paragraphs: list[dict[str, Any]], max_chars: int = 900) -> list[dict[str, Any]]:
    compact = []
    for paragraph in paragraphs:
        text = str(paragraph.get("paragraph_text", ""))
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        compact.append(
            {
                "idx": paragraph.get("idx"),
                "title": paragraph.get("title", ""),
                "text": text,
                "is_supporting": paragraph.get("is_supporting"),
            }
        )
    return compact




def write_failed_response(args: argparse.Namespace, output_index: int, attempt: int, text: str, error: Exception) -> None:
    if not args.failed_response_dir:
        return
    failure_dir = Path(args.failed_response_dir)
    failure_dir.mkdir(parents=True, exist_ok=True)
    safe_error = re.sub(r"[^A-Za-z0-9_.-]+", "_", type(error).__name__)[:80]
    path = failure_dir / f"musique_{output_index:05d}_attempt_{attempt + 1}_{safe_error}.txt"
    preview = [
        f"source_index={output_index}",
        f"attempt={attempt + 1}",
        f"error={type(error).__name__}: {error}",
        "",
        text,
    ]
    path.write_text("\n".join(preview), encoding="utf-8", errors="replace")


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


def convert_item_with_llm(item: dict[str, Any], output_index: int, args: argparse.Namespace) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.llm_retries + 1):
        response_text = ""
        try:
            if attempt:
                print(f"Retrying source index {output_index}, attempt {attempt + 1}/{args.llm_retries + 1}")
            response_text = call_llm(build_llm_prompt(item, output_index, args.paragraph_max_chars), args)
            dag = parse_json_object(response_text)
            validate_dag_shape(dag)
            dag["dag_id"] = f"musique_{output_index + 1:04d}"
            dag.setdefault("metadata", {})
            dag["metadata"]["source"] = "MuSiQue"
            dag["metadata"]["source_id"] = item.get("id", "")
            dag["metadata"]["source_index"] = output_index
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
            if response_text:
                write_failed_response(args, output_index, attempt, response_text, exc)
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


def read_index_file(path: str) -> set[int]:
    if not path:
        return set()
    index_path = Path(path)
    if not index_path.exists():
        return set()
    indices: set[int] = set()
    for raw_line in index_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            indices.add(int(line.split()[0]))
        except ValueError:
            continue
    return indices


def write_index_file(path: str, indices: set[int]) -> None:
    if not path:
        return
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(str(index) for index in sorted(indices))
    index_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def record_failed_indices(path: str, failed_indices: set[int]) -> None:
    if not path or not failed_indices:
        return
    indices = read_index_file(path)
    indices.update(failed_indices)
    write_index_file(path, indices)


def convert_single(item: dict[str, Any], item_index: int, args: argparse.Namespace, use_llm: bool) -> dict[str, Any]:
    return convert_item_with_llm(item, item_index, args) if use_llm else convert_item(item, item_index)


def convert(args: argparse.Namespace, output_path: Path) -> list[dict[str, Any]]:
    all_items = iter_jsonl(Path(args.input))
    end_index = min(len(all_items), args.start + args.limit)
    requested_indices = set(range(args.start, end_index))
    target_indices = set(requested_indices)
    deferred_failed_indices = read_index_file(args.failed_index_file)
    if args.skip_failed_indices and deferred_failed_indices:
        target_indices -= deferred_failed_indices
    if args.retry_failed_indices:
        target_indices &= deferred_failed_indices
    use_llm = args.converter == "llm" or args.use_llm

    dags_by_index: dict[int, dict[str, Any]] = {}
    existing_non_llm = 0
    existing_outside_range = 0
    for dag in load_existing_output(output_path):
        source_index = get_source_index(dag)
        if source_index in requested_indices:
            metadata = dag.get("metadata") if isinstance(dag.get("metadata"), dict) else {}
            conversion = str(metadata.get("conversion", "")).strip()
            if source_index in target_indices and conversion != "llm_autonomous":
                existing_non_llm += 1
                if args.replace_non_llm:
                    continue
            dags_by_index[source_index] = dag
        elif source_index is not None:
            existing_outside_range += 1

    if existing_outside_range and not args.drop_existing_outside_range:
        raise RuntimeError(
            f"Existing output contains {existing_outside_range} DAGs outside requested range "
            f"{args.start}:{end_index}. Increase --limit/--start to include them, or pass "
            "--drop-existing-outside-range to intentionally rewrite only this range."
        )

    if dags_by_index:
        print(f"Resuming from existing output with {len(dags_by_index)} completed DAGs: {output_path}")
    if existing_non_llm:
        if args.replace_non_llm:
            print(f"Rebuilding {existing_non_llm} non-LLM DAGs from existing output: {output_path}")
            write_output(output_path, sorted_dags(dags_by_index))
        elif use_llm:
            print(
                f"Existing output contains {existing_non_llm} non-LLM DAGs; "
                "rerun with --replace-non-llm to regenerate them."
            )

    pending = [(index, all_items[index]) for index in sorted(target_indices) if index not in dags_by_index]
    if args.max_pending > 0:
        pending = pending[: args.max_pending]
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
    print(f"Using {max_workers} workers for {len(pending)} pending MuSiQue conversions")
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
        failed_index_set = set(failures)
        if args.retry_failed_indices:
            write_index_file(args.failed_index_file, {index for index in target_indices if index not in dags_by_index})
        else:
            record_failed_indices(args.failed_index_file, failed_index_set)
        write_output(output_path, sorted_dags(dags_by_index))
        failed_indices = ", ".join(str(index) for index in sorted(failures)[:20])
        extra = "" if len(failures) <= 20 else f", ... +{len(failures) - 20} more"
        if args.ignore_failures:
            print(
                f"Deferred {len(failures)} failed MuSiQue conversions for later retry. "
                f"Failed source indices: {failed_indices}{extra}"
            )
            return sorted_dags(dags_by_index)
        raise RuntimeError(
            f"{len(failures)} MuSiQue conversions failed; checkpoint saved to {output_path}. "
            f"Failed source indices: {failed_indices}{extra}"
        )

    if args.retry_failed_indices:
        write_index_file(args.failed_index_file, {index for index in target_indices if index not in dags_by_index})
    return sorted_dags(dags_by_index)
