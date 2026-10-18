from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest


SCRIPT = Path(__file__).resolve().parents[2]
CLI = SCRIPT / "convert_dag.py"
DATASETS = ("strategyqa", "musique", "toolbench", "alfworld")


def run_cli(*arguments, cwd, cli=CLI):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-I", "-B", str(cli), *map(str, arguments)],
        cwd=cwd, capture_output=True, text=True, timeout=30, env=env,
    )


def run_module(module, *arguments):
    return subprocess.run(
        [sys.executable, "-B", "-m", "src." + module, *map(str, arguments)],
        cwd=SCRIPT, capture_output=True, text=True, timeout=60,
    )


def make_source(root, dataset):
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "question": "Where is the cup?", "answer": "Table",
        "facts": ["The cup is on the table."],
        "decomposition": ["Locate the cup."],
        "paragraphs": [{"idx": 0, "title": "Cup", "paragraph_text": "On the table.", "is_supporting": True}],
        "question_decomposition": [{"id": 1, "question": "Locate the cup.", "paragraph_support_idx": 0}],
        "conversations": [
            {"from": "system", "value": "Tools: {'name': 'locate'}"},
            {"from": "user", "value": "Find the cup."},
            {"from": "assistant", "value": "Action: locate\nAction Input: cup"},
            {"from": "function", "value": "The cup is on the table."},
        ],
    }
    if dataset == "alfworld":
        source = root / "json_2.1.1"
        for index in range(2):
            path = source / "train" / f"pick_and_place_simple-{index}" / "trial" / "traj_data.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "task_type": "pick_and_place_simple",
                "pddl_params": {"object_target": "Cup", "parent_target": "Table"},
                "turk_annotations": {"anns": [{"task_desc": f"Place cup {index} on the table."}]},
                "plan": {"high_pddl": [{"discrete_action": {"action": "PickupObject", "args": ["Cup"]}}]},
            }), encoding="utf-8")
        return "--input-root", source
    records = [{**record, "id": index, "qid": str(index)} for index in range(2)]
    source = root / ("toolllama_G123_dfs_train.json" if dataset == "toolbench" else "train.jsonl")
    content = json.dumps(records) if dataset == "toolbench" else "\n".join(map(json.dumps, records)) + "\n"
    source.write_text(content, encoding="utf-8")
    return "--input", source


def read_pair(output):
    graphs = json.loads(output.read_text(encoding="utf-8"))
    sidecar = output.with_name(output.stem + "_resource_estimates.json")
    resources = json.loads(sidecar.read_text(encoding="utf-8"))
    assert [g["dag_id"] for g in graphs] == [g["dag_id"] for g in resources["dags"]]
    for graph, resource in zip(graphs, resources["dags"]):
        ids = {n["node_id"] for n in graph["nodes"]}
        assert len(ids) == len(graph["nodes"])
        assert ids == {n["node_id"] for n in resource["nodes"]}
        assert all(n["node_type"] in {"CPU", "GPU"} and n["prompt"] for n in graph["nodes"])
        assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    return graphs, resources


def test_entry_help_outside_checkout(tmp_path):
    result = run_cli("--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout


@pytest.mark.parametrize("dataset", DATASETS)
def test_rule_conversion_and_offset_outside_checkout(tmp_path, dataset):
    option, source = make_source(tmp_path / "input", dataset)
    output = tmp_path / dataset / "dags.json"
    result = run_cli("--dataset", dataset, option, source, "--output", output,
                     "--converter", "rule", "--start", "1", "--limit", "0", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    graphs, _ = read_pair(output)
    assert len(graphs) == 1
    assert graphs[0]["metadata"]["source_index"] == 1


@pytest.mark.parametrize("dataset", DATASETS)
def test_default_paths_are_relative_to_argb_r_not_working_directory(tmp_path, dataset):
    destination = tmp_path / "copied" / "ARGB-R" / "script"
    shutil.copytree(SCRIPT, destination, ignore=shutil.ignore_patterns("__pycache__", "tests", ".pytest_cache"))
    make_source(destination.parent / "raw" / dataset, dataset)
    result = run_cli("--dataset", dataset, "--converter", "rule", "--limit", "1",
                     cwd=tmp_path, cli=destination / "convert_dag.py")
    assert result.returncode == 0, result.stdout + result.stderr
    graphs, _ = read_pair(destination.parent / "generated" / dataset / "dags.json")
    assert len(graphs) == 1
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("option,value", [("--start", "-1"), ("--limit", "-1"), ("--workers", "0"), ("--llm-retries", "-1")])
def test_invalid_counts_fail_before_reading_input(tmp_path, option, value):
    result = run_cli("--dataset", "strategyqa", option, value, cwd=tmp_path)
    assert result.returncode == 2
    assert "must be" in result.stderr


def test_empty_toolbench_has_empty_graph_and_resource_arrays(tmp_path):
    source = tmp_path / "empty.json"
    source.write_text("[]", encoding="utf-8")
    output = tmp_path / "output.json"
    result = run_cli("--dataset", "toolbench", "--input", source, "--output", output,
                     "--converter", "rule", "--limit", "0", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    graphs, resources = read_pair(output)
    assert graphs == resources["dags"] == []


@contextmanager
def model_endpoint(fail_first=False):
    graph = {
        "dag_id": "model-generated", "original_question": "Place the cup on the table.",
        "nodes": [
            {"node_id": "N0", "node_type": "GPU", "prompt": "Plan the task.",
             "metadata": {"operation_type": "planning", "device_reason": "Requires model inference."}},
            {"node_id": "N1", "node_type": "CPU", "prompt": "Locate the cup.",
             "metadata": {"operation_type": "tool_call", "device_reason": "Invokes the environment API."}},
            {"node_id": "N2", "node_type": "CPU", "prompt": "Place the cup on the table.",
             "metadata": {"operation_type": "environment_action", "device_reason": "Invokes the environment API."}},
            {"node_id": "N3", "node_type": "GPU", "prompt": "Summarize completion.",
             "metadata": {"operation_type": "answer_generation", "device_reason": "Requires model inference."}},
        ],
        "edges": [{"source": "N0", "target": "N1"}, {"source": "N1", "target": "N2"}, {"source": "N2", "target": "N3"}],
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if fail_first and len(requests) == 1:
                self.send_response(503)
                self.end_headers()
                return
            body = json.dumps({"choices": [{"message": {"content": json.dumps(graph)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("dataset", DATASETS)
def test_llm_conversion_with_concurrency_over_local_http(tmp_path, dataset):
    option, source = make_source(tmp_path / "input", dataset)
    output = tmp_path / "generated" / "dags.json"
    with model_endpoint() as (url, requests):
        result = run_cli("--dataset", dataset, option, source, "--output", output,
                         "--converter", "llm", "--llm-base-url", url,
                         "--api-key", "test-key", "--limit", "0", "--workers", "2", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    graphs, _ = read_pair(output)
    assert len(graphs) == 2
    assert [g["metadata"]["source_index"] for g in graphs] == [0, 1]
    assert all(len(g["nodes"]) == 4 and len(g["edges"]) == 3 for g in graphs)
    assert all(g["metadata"]["conversion"].startswith("llm") for g in graphs)


def test_transient_model_error_retries_and_completes(tmp_path):
    option, source = make_source(tmp_path / "input", "strategyqa")
    output = tmp_path / "output.json"
    with model_endpoint(fail_first=True) as (url, _):
        result = run_cli("--dataset", "strategyqa", option, source, "--output", output,
                         "--llm-base-url", url, "--api-key", "test-key", "--limit", "1",
                         "--llm-retries", "1", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(read_pair(output)[0]) == 1


def test_checkpoint_resume_does_not_duplicate_strategyqa_graphs(tmp_path):
    option, source = make_source(tmp_path / "input", "strategyqa")
    output = tmp_path / "output.json"
    args = ("--dataset", "strategyqa", option, source, "--output", output, "--converter", "rule", "--limit", "0")
    for _ in range(2):
        result = run_cli(*args, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
    assert len(read_pair(output)[0]) == 2


@pytest.mark.parametrize("module", ["fetch_raw_datasets", "calibrate_cpu_resource_estimates", "verify_data"])
def test_auxiliary_help_does_not_require_original_checkout_or_download_dependencies(module):
    result = run_module(module, "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


def test_cpu_calibration_preserves_graphs_and_initial_sidecar(tmp_path):
    option, source = make_source(tmp_path / "input", "strategyqa")
    output = tmp_path / "dags.json"
    result = run_cli("--dataset", "strategyqa", option, source, "--output", output,
                     "--converter", "rule", "--limit", "1", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    _, resources = read_pair(output)
    result = run_module("calibrate_cpu_resource_estimates", output)
    assert result.returncode == 0, result.stdout + result.stderr
    calibrated = output.with_name("dags_resource_estimates_local_sim_v2.json")
    payload = json.loads(calibrated.read_text(encoding="utf-8"))
    assert output.read_bytes() == original
    assert read_pair(output)[1] == resources
    assert payload["metadata"]["source_dag_sha256"] == hashlib.sha256(original).hexdigest()
    cpus = [n["cpu"] for g in payload["dags"] for n in g["nodes"] if n["node_type"] == "CPU"]
    assert cpus and all(n["profile"] == "local_sim_v2" and n["time_ms"] > 0 for n in cpus)
    assert run_module("calibrate_cpu_resource_estimates", output).returncode != 0
    assert run_module("calibrate_cpu_resource_estimates", output, "--force").returncode == 0
