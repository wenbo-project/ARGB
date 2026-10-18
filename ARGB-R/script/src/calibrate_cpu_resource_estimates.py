"""Generate operation-aware CPU resource sidecars for existing DAG files."""

from __future__ import annotations



import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .dag_standard import LOCAL_SIM_CPU_PROFILE, build_resource_estimates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-estimate CPU node durations without changing DAG topology, node types, "
            "or legacy resource sidecars."
        )
    )
    parser.add_argument("dag_files", nargs="+", type=Path)
    parser.add_argument(
        "--profile",
        choices=(LOCAL_SIM_CPU_PROFILE,),
        default=LOCAL_SIM_CPU_PROFILE,
    )
    parser.add_argument(
        "--include-node-context",
        action="store_true",
        help="Copy prompts and node metadata into the sidecar; disabled by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing sidecar for the same profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dag_file in args.dag_files:
        output_file = calibrated_resource_path(dag_file, args.profile)
        if output_file.exists() and not args.force:
            raise FileExistsError(
                f"{output_file} already exists; rerun with --force to replace it"
            )

        dags = load_dags(dag_file)
        payload = build_resource_estimates(
            dags,
            cpu_estimation_profile=args.profile,
            include_node_context=args.include_node_context,
        )
        payload["metadata"].update(
            {
                "generated_by": "ARGB-R/script/src/calibrate_cpu_resource_estimates.py",
                "source_dag_file": dag_file.as_posix(),
                "source_dag_sha256": sha256_file(dag_file),
            }
        )
        write_json_atomic(output_file, payload)
        cpu_nodes, old_total_ms, new_total_ms = cpu_duration_summary(dags, payload)
        print(
            f"{dag_file}: {cpu_nodes} CPU nodes, "
            f"{old_total_ms} ms -> {new_total_ms} ms; wrote {output_file}"
        )


def calibrated_resource_path(dag_file: Path, profile: str) -> Path:
    return dag_file.with_name(
        f"{dag_file.stem}_resource_estimates_{profile}{dag_file.suffix}"
    )


def load_dags(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    dags = payload.get("dags", []) if isinstance(payload, dict) else payload
    if not isinstance(dags, list):
        raise ValueError(f"expected a DAG list in {path}")
    return dags


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def cpu_duration_summary(
    dags: list[dict[str, Any]],
    resource_payload: dict[str, Any],
) -> tuple[int, int, int]:
    old_total_ms = 0
    cpu_nodes = 0
    for dag in dags:
        for node in dag.get("nodes", []):
            if str(node.get("node_type", "")).strip().lower() not in {
                "cpu",
                "tool_call",
            }:
                continue
            cpu_nodes += 1
            cpu = node.get("cpu") if isinstance(node.get("cpu"), dict) else {}
            old_total_ms += int(cpu.get("time_ms", 0) or 0)

    new_total_ms = sum(
        int((node.get("cpu") or {}).get("time_ms", 0) or 0)
        for dag in resource_payload.get("dags", [])
        for node in dag.get("nodes", [])
        if str(node.get("node_type", "")).strip().lower() == "cpu"
    )
    return cpu_nodes, old_total_ms, new_total_ms


if __name__ == "__main__":
    main()
