from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def release_fixture(tmp_path, sharded=False):
    root = tmp_path / "ARGB-R"
    graph = {
        "dag_id": "example_0001",
        "nodes": [
            {"node_id": "N0", "node_type": "GPU", "prompt": "Reason about the task."},
            {"node_id": "N1", "node_type": "CPU", "prompt": "Format the result."},
        ],
        "edges": [{"source": "N0", "target": "N1"}],
    }
    graph_path = root / "example" / ("shard-1.json" if sharded else "dags.json")
    resource_path = root / "example/resources.json"
    write_json(graph_path, [graph])
    write_json(resource_path, {"dags": [copy.deepcopy(graph)]})
    input_path = graph_path
    if sharded:
        input_path = root / "example/dags.index.json"
        write_json(input_path, {"dags": 1, "shards": [{"path": graph_path.name, "dags": 1, "sha256": sha(graph_path)}]})
    dataset = {
        "name": "example", "dag_file": str(input_path.relative_to(tmp_path)).replace("\\", "/"),
        "resource_estimate_file": "ARGB-R/example/resources.json",
        "sha256": {"dag_file": sha(input_path), "resource_estimate_file": sha(resource_path)},
        "dags": 1, "nodes": 2, "edges": 1, "gpu_nodes": 1, "cpu_nodes": 1, "average_depth": 2,
    }
    if sharded:
        dataset["dag_shards"] = [{"path": str(graph_path.relative_to(tmp_path)).replace("\\", "/"), "dags": 1, "sha256": sha(graph_path)}]
    manifest = {
        "datasets": [dataset],
        "paper_statistics": {
            "dags": 1, "nodes": 2, "edges": 1, "gpu_nodes": 1, "cpu_nodes": 1,
            "mean_nodes_per_dag": 2, "node_count_population_std": 0,
            "mean_depth": 2, "depth_population_std": 0,
        },
    }
    write_json(root / "manifest.json", manifest)
    return root, manifest, graph_path, resource_path


def verify(root):
    return subprocess.run([sys.executable, "-B", "-m", "src.verify_data", "--data-root", str(root)],
                          cwd=SCRIPT, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("sharded", [False, True])
def test_valid_release_passes_and_keeps_data_unchanged(tmp_path, sharded):
    root, _, _, _ = release_fixture(tmp_path, sharded)
    before = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    result = verify(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 graphs" in result.stdout
    assert before == {str(p): p.read_bytes() for p in root.rglob("*.json")}


@pytest.mark.parametrize("sharded", [False, True])
def test_release_moved_into_datasets_keeps_legacy_manifest_paths(tmp_path, sharded):
    root, _, _, _ = release_fixture(tmp_path, sharded)
    originals = list(root.iterdir())
    nested = root / "datasets"
    nested.mkdir()
    for path in originals:
        path.rename(nested / path.name)
    for supplied_root in [root, nested]:
        result = verify(supplied_root)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 graphs" in result.stdout


@pytest.mark.parametrize("failure", ["hash", "endpoint", "cycle", "duplicate_node", "resource_pair", "counts", "shard_hash"])
def test_corrupt_release_is_rejected(tmp_path, failure):
    root, manifest, graph_path, resource_path = release_fixture(tmp_path, failure == "shard_hash")
    graphs = json.loads(graph_path.read_text(encoding="utf-8"))
    dataset = manifest["datasets"][0]
    if failure in {"hash", "shard_hash"}:
        graph_path.write_bytes(graph_path.read_bytes() + b" ")
    elif failure == "resource_pair":
        resources = json.loads(resource_path.read_text(encoding="utf-8"))
        resources["dags"][0]["nodes"].pop()
        write_json(resource_path, resources)
        dataset["sha256"]["resource_estimate_file"] = sha(resource_path)
    elif failure == "counts":
        dataset["nodes"] = 3
    else:
        if failure == "endpoint":
            graphs[0]["edges"][0]["target"] = "missing"
        elif failure == "cycle":
            graphs[0]["edges"].append({"source": "N1", "target": "N0"})
        elif failure == "duplicate_node":
            graphs[0]["nodes"][1]["node_id"] = "N0"
        write_json(graph_path, graphs)
        dataset["sha256"]["dag_file"] = sha(graph_path)
    write_json(root / "manifest.json", manifest)
    assert verify(root).returncode != 0
