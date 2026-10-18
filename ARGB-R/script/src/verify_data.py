"""Validate the released ARGB-R files without importing a scheduling package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


COUNT_KEYS = ("dags", "nodes", "edges", "gpu_nodes", "cpu_nodes")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash(path: Path, expected: str) -> None:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def iter_graphs(path: Path):
    payload = read_json(path)
    if isinstance(payload, list):
        yield from payload
        return
    if not isinstance(payload, dict) or not isinstance(payload.get("shards"), list):
        raise ValueError(f"Expected a graph array or shard index: {path}")
    count = 0
    for shard in payload["shards"]:
        shard_path = path.parent / shard["path"]
        check_hash(shard_path, shard["sha256"])
        graphs = read_json(shard_path)
        if not isinstance(graphs, list) or len(graphs) != shard["dags"]:
            raise ValueError(f"Shard count mismatch: {shard_path}")
        yield from graphs
        count += len(graphs)
    if count != payload["dags"]:
        raise ValueError(f"Index count mismatch: {path}")


def graph_info(graph: dict) -> tuple[dict, int]:
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError(f"Invalid nodes or edges: {graph.get('dag_id')}")
    ids = [node["node_id"] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate node identifiers: {graph['dag_id']}")
    if any(node.get("node_type") not in {"CPU", "GPU"} for node in nodes):
        raise ValueError(f"Invalid resource label: {graph['dag_id']}")
    if any(not isinstance(node.get("prompt"), str) or not node["prompt"].strip() for node in nodes):
        raise ValueError(f"Missing execution prompt: {graph['dag_id']}")
    remaining = dict.fromkeys(ids, 0)
    successors = {node_id: [] for node_id in ids}
    pairs = set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source not in remaining or target not in remaining:
            raise ValueError(f"Edge references unknown node: {graph['dag_id']}")
        if (source, target) in pairs:
            raise ValueError(f"Duplicate edge: {graph['dag_id']}")
        pairs.add((source, target))
        successors[source].append(target)
        remaining[target] += 1
    ready = [node_id for node_id in ids if remaining[node_id] == 0]
    depth = {node_id: 1 for node_id in ready}
    for node_id in ready:
        for target in successors[node_id]:
            depth[target] = max(depth.get(target, 1), depth[node_id] + 1)
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
    if len(ready) != len(nodes):
        raise ValueError(f"Graph is not acyclic: {graph['dag_id']}")
    return {
        "dags": 1, "nodes": len(nodes), "edges": len(edges),
        "gpu_nodes": sum(node["node_type"] == "GPU" for node in nodes),
        "cpu_nodes": sum(node["node_type"] == "CPU" for node in nodes),
    }, max(depth.values())


def check_resource_pair(graph: dict, resource: dict) -> None:
    graph_nodes = {node["node_id"]: node["node_type"] for node in graph["nodes"]}
    resource_nodes = {node["node_id"]: node["node_type"] for node in resource["nodes"]}
    if len(resource_nodes) != len(resource["nodes"]) or graph_nodes != resource_nodes:
        raise ValueError(f"Resource nodes do not match: {graph['dag_id']}")
    graph_edges = {(edge["source"], edge["target"]) for edge in graph["edges"]}
    resource_edges = {(edge["source"], edge["target"]) for edge in resource["edges"]}
    if len(resource_edges) != len(resource["edges"]) or graph_edges != resource_edges:
        raise ValueError(f"Resource edges do not match: {graph['dag_id']}")


def resolve_release_path(data_root: Path, filename: str) -> Path:
    path = Path(filename)
    candidates = [data_root.parent / path, data_root / path]
    parts = list(path.parts)
    while parts and parts[0] in {"ARGB-R", "datasets"}:
        parts.pop(0)
    candidates.append(data_root.joinpath(*parts))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Release file not found: {filename} under {data_root}")


def verify_release(data_root: Path) -> dict:
    data_root = data_root.resolve()
    if not (data_root / "manifest.json").is_file() and (data_root / "datasets/manifest.json").is_file():
        data_root = data_root / "datasets"
    release = read_json(data_root / "manifest.json")
    all_counts = []
    all_depths = []
    totals = dict.fromkeys(COUNT_KEYS, 0)
    seen_ids = set()
    for dataset in release["datasets"]:
        graph_path = resolve_release_path(data_root, dataset["dag_file"])
        resource_path = resolve_release_path(data_root, dataset["resource_estimate_file"])
        check_hash(graph_path, dataset["sha256"]["dag_file"])
        check_hash(resource_path, dataset["sha256"]["resource_estimate_file"])
        if dataset.get("dag_shards"):
            index = read_json(graph_path)
            actual_shards = {
                (graph_path.parent / shard["path"]).resolve(): (shard["dags"], shard["sha256"])
                for shard in index["shards"]
            }
            declared_shards = {
                resolve_release_path(data_root, shard["path"]): (shard["dags"], shard["sha256"])
                for shard in dataset["dag_shards"]
            }
            if actual_shards != declared_shards:
                raise ValueError(f"Manifest and shard index differ: {dataset['name']}")
        resource_list = read_json(resource_path)["dags"]
        resources = {graph["dag_id"]: graph for graph in resource_list}
        if len(resources) != len(resource_list):
            raise ValueError(f"Duplicate resource graph identifiers: {dataset['name']}")
        counts = dict.fromkeys(COUNT_KEYS, 0)
        depths = []
        dataset_ids = set()
        for graph in iter_graphs(graph_path):
            graph_id = graph["dag_id"]
            if graph_id in seen_ids:
                raise ValueError(f"Duplicate graph identifier: {graph_id}")
            seen_ids.add(graph_id)
            dataset_ids.add(graph_id)
            info, depth = graph_info(graph)
            if graph_id not in resources:
                raise ValueError(f"Missing resource graph: {graph_id}")
            check_resource_pair(graph, resources[graph_id])
            for key in COUNT_KEYS:
                counts[key] += info[key]
            depths.append(depth)
            all_depths.append(depth)
            all_counts.append(info["nodes"])
        if dataset_ids != set(resources):
            raise ValueError(f"Unmatched resource graphs: {dataset['name']}")
        for key in COUNT_KEYS:
            if counts[key] != dataset[key]:
                raise ValueError(f"Inventory mismatch for {dataset['name']}: {key}")
            totals[key] += counts[key]
        if not depths or not math.isclose(statistics.fmean(depths), dataset["average_depth"], abs_tol=1e-9):
            raise ValueError(f"Average depth mismatch: {dataset['name']}")
        print(f"{dataset['name']}: {counts['dags']:,} graphs, {counts['nodes']:,} nodes, {counts['edges']:,} edges OK")
    paper = release["paper_statistics"]
    for key in COUNT_KEYS:
        if totals[key] != paper[key]:
            raise ValueError(f"Aggregate mismatch: {key}")
    statistics_to_check = {
        "mean_nodes_per_dag": statistics.fmean(all_counts),
        "node_count_population_std": statistics.pstdev(all_counts),
        "mean_depth": statistics.fmean(all_depths),
        "depth_population_std": statistics.pstdev(all_depths),
    }
    for key, actual in statistics_to_check.items():
        if not math.isclose(actual, paper[key], abs_tol=1e-9):
            raise ValueError(f"Aggregate statistics mismatch: {key}")
    print(f"Verified {totals['dags']:,} graphs / {totals['nodes']:,} nodes / {totals['edges']:,} edges")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARGB-R release hashes, graph structure, resources, and statistics.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    verify_release(args.data_root)


if __name__ == "__main__":
    main()
