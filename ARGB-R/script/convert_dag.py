"""Unified ARGB-R converter for the four released real datasets.

The dataset implementations are private backends in src/.  Users
should invoke this module as the single conversion CLI so every dataset gets
the same path resolution, checkpoint, standardization, and resource-sidecar
workflow.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT))

from src.dag_conversion_common import load_dotenv
from src.dag_standard import build_resource_estimates, resource_estimate_path, standardize_dags


DATASET_DEFAULTS = {
    "strategyqa": {
        "input": str(DATA_ROOT / "raw/strategyqa/train.jsonl"),
        "output": str(DATA_ROOT / "generated/strategyqa/dags.json"),
    },
    "musique": {
        "input": str(DATA_ROOT / "raw/musique/train.jsonl"),
        "output": str(DATA_ROOT / "generated/musique/dags.json"),
    },
    "toolbench": {
        "input": str(DATA_ROOT / "raw/toolbench/toolllama_G123_dfs_train.json"),
        "output": str(DATA_ROOT / "generated/toolbench/dags.json"),
    },
    "alfworld": {
        "input_root": str(DATA_ROOT / "raw/alfworld/json_2.1.1"),
        "output": str(DATA_ROOT / "generated/alfworld/dags.json"),
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the common CLI and the small set of useful advanced options."""

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Convert an ARGB-R dataset into standardized DAG and resource JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DATASET_DEFAULTS),
        help="ARGB-R dataset to convert.",
    )
    parser.add_argument("--input", help="Input JSONL/JSON file (not used for ALFWorld).")
    parser.add_argument("--input-root", help="ALFWorld root containing split directories.")
    parser.add_argument("--splits", nargs="+", default=None, help="ALFWorld splits to scan.")
    parser.add_argument("--output", help="Output DAG JSON path.")
    parser.add_argument("--limit", type=int, default=100, help="Number of DAGs; 0 means all available records.")
    parser.add_argument("--start", type=int, default=0, help="Source-record offset.")
    parser.add_argument(
        "--converter",
        choices=("llm", "rule"),
        default="llm",
        help="Conversion backend shared by all four datasets.",
    )
    parser.add_argument(
        "--allow-rule-fallback",
        action="store_true",
        help="Use the dataset rule converter when an LLM conversion fails (where supported).",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--llm-api", choices=("chat", "completion"), default="chat")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)

    # Dataset-specific controls share this CLI; private backends own parsing
    # and conversion of their respective source formats.
    parser.add_argument("--paragraph-max-chars", type=int, default=900)
    parser.add_argument("--observation-chars", type=int, default=500)
    parser.add_argument("--thinking", choices=("none", "enabled", "disabled"), default="none")
    parser.add_argument("--reasoning-effort", choices=("", "low", "medium", "high"), default="")
    parser.add_argument("--require-nonlinear", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-pending", type=int, default=0)
    parser.add_argument("--sample-mode", choices=("balanced", "first"), default="balanced")
    parser.add_argument("--max-consecutive-failures", type=int, default=8)
    parser.add_argument(
        "--skip-invalid-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip invalid ALFWorld/ToolBench LLM results after retries.",
    )
    args = parser.parse_args(argv)
    if args.start < 0 or args.limit < 0:
        parser.error("--start and --limit must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.llm_retries < 0:
        parser.error("--llm-retries must be non-negative")
    return args


def _read_record_count(path: Path, dataset: str) -> int | None:
    """Return a JSON/JSONL record count so ``--limit 0`` means all records."""

    if dataset == "alfworld":
        return None
    if dataset in {"strategyqa", "musique"}:
        with path.open("r", encoding="utf-8") as handle:
            return sum(bool(line.strip()) for line in handle)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return len(data)


def _resolve_paths(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.dataset == "alfworld":
        # ``--input`` is accepted as a convenient alias for an ALFWorld root,
        # while ``--input-root`` remains the explicit spelling.
        args.input_root = args.input_root or args.input or defaults["input_root"]
        args.splits = args.splits or ["train"]
    else:
        args.input = args.input or defaults["input"]
        if args.limit == 0:
            count = _read_record_count(Path(args.input), args.dataset)
            args.limit = max(count - args.start, 0) if count is not None else 0
    args.output = args.output or defaults["output"]
    if args.start < 0 or args.limit < 0:
        raise ValueError("--start and --limit must be non-negative")


def _load_converter(dataset: str) -> ModuleType:
    module_name = {
        "strategyqa": "_strategyqa",
        "musique": "_musique",
        "toolbench": "_toolbench",
        "alfworld": "_alfworld",
    }[dataset]
    return import_module(f"src.{module_name}")


def _converter_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Supply the common settings and internal backend controls."""

    values: dict[str, Any] = vars(args).copy()
    values["use_llm"] = args.converter == "llm"
    values["no_rule_fallback"] = not args.allow_rule_fallback
    values["input"] = getattr(args, "input", None)
    values["input_root"] = getattr(args, "input_root", None)
    # The modules expect strings for these optional paths.
    values.setdefault("failed_response_dir", "")
    values.setdefault("failed_index_file", "")
    values.setdefault("skip_failed_indices", False)
    values.setdefault("retry_failed_indices", False)
    values.setdefault("replace_non_llm", False)
    values.setdefault("drop_existing_outside_range", False)
    values.setdefault("ignore_failures", False)
    return argparse.Namespace(**values)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def convert(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Dispatch conversion and return standardized DAGs."""

    _resolve_paths(args)
    module = _load_converter(args.dataset)
    module_args = _converter_namespace(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "toolbench":
        dags = module.convert(Path(args.input), output_path, module_args)
    else:
        dags = module.convert(module_args, output_path)
    standardized = standardize_dags(dags)
    _write_json_atomic(output_path, standardized)
    _write_json_atomic(resource_estimate_path(output_path), build_resource_estimates(standardized))
    return standardized


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dags = convert(args)
    print(
        f"Wrote {len(dags)} {args.dataset} DAGs, "
        f"{sum(len(dag.get('nodes', [])) for dag in dags)} nodes, "
        f"{sum(len(dag.get('edges', [])) for dag in dags)} edges to {args.output}"
    )


if __name__ == "__main__":
    main()
