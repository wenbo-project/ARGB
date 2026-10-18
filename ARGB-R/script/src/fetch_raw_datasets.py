"""Fetch the three Hugging Face sources used by ARGB-R converters.

The script stores each split as JSONL under ARGB-R/raw/<dataset_name>/ and
writes a manifest with source IDs, configs, splits, row counts, and local paths.
ALFWorld trajectories must be obtained separately from the official release.
"""

from __future__ import annotations



import argparse
import fnmatch
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable



@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    config: str | None
    splits: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class RawFileSpec:
    key: str
    hf_id: str
    files: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class SnapshotSpec:
    key: str
    hf_id: str
    allow_patterns: tuple[str, ...]
    note: str


DATASETS: dict[str, DatasetSpec] = {
    "musique": DatasetSpec(
        key="musique",
        hf_id="dgslibisey/MuSiQue",
        config=None,
        splits=("train", "validation"),
        note="MuSiQue multi-hop QA.",
    ),
}


RAW_FILE_DATASETS: dict[str, RawFileSpec] = {
    "strategyqa": RawFileSpec(
        key="strategyqa",
        hf_id="njf/StrategyQA",
        files=(
            "train.jsonl",
            "dev.jsonl",
            "dev_decomp_3_plus.jsonl",
            "dev_decomp_4_plus.jsonl",
            "dev_decomp_max.jsonl",
        ),
        note="StrategyQA yes/no multi-hop reasoning files.",
    ),
}


SNAPSHOT_DATASETS: dict[str, SnapshotSpec] = {
    "toolbench": SnapshotSpec(
        key="toolbench",
        hf_id="Adorg/ToolBench",
        allow_patterns=(
            "README.md",
            "toolllama_G123_dfs_train.json",
            "toolllama_G123_dfs_eval.json",
            "instruction/*.json",
            "retrieval/G1/*",
            "test_query_ids/*.json",
        ),
        note=(
            "ToolBench core raw data. The full repo has 33k+ toolenv/answer "
            "files; this fetch keeps the core train/eval/instruction/retrieval files."
        ),
    ),
}

ALL_DATASETS = {**DATASETS, **RAW_FILE_DATASETS, **SNAPSHOT_DATASETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "raw"),
        help="Directory where raw JSONL files and manifest are written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=tuple(ALL_DATASETS),
        choices=tuple(ALL_DATASETS),
        help="Dataset keys to fetch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows per split for smoke tests.",
    )
    return parser.parse_args()


def iter_specs(keys: Iterable[str]) -> Iterable[DatasetSpec]:
    for key in keys:
        if key in DATASETS:
            yield DATASETS[key]


def iter_raw_file_specs(keys: Iterable[str]) -> Iterable[RawFileSpec]:
    for key in keys:
        if key in RAW_FILE_DATASETS:
            yield RAW_FILE_DATASETS[key]


def iter_snapshot_specs(keys: Iterable[str]) -> Iterable[SnapshotSpec]:
    for key in keys:
        if key in SNAPSHOT_DATASETS:
            yield SNAPSHOT_DATASETS[key]


def split_output_path(output_dir: Path, spec: DatasetSpec, split: str) -> Path:
    return output_dir / spec.key / f"{split}.jsonl"


def fetch_split(spec: DatasetSpec, split: str, output_dir: Path, limit: int | None) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(
        spec.hf_id,
        spec.config,
        split=split,
        trust_remote_code=False,
    )
    original_rows = len(dataset)
    if limit is not None:
        dataset = dataset.select(range(min(limit, original_rows)))

    out_path = split_output_path(output_dir, spec, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(out_path), orient="records", lines=True, force_ascii=False)

    return {
        "dataset": spec.key,
        "hf_id": spec.hf_id,
        "config": spec.config,
        "split": split,
        "rows": len(dataset),
        "original_rows": original_rows,
        "path": str(out_path.as_posix()),
        "note": spec.note,
    }


def count_jsonl_lines(path: Path, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for count, _ in enumerate(f, start=1):
            if limit is not None and count >= limit:
                break
    return count


def copy_limited_jsonl(source: Path, target: Path, limit: int) -> int:
    rows = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for rows, line in enumerate(src, start=1):
            if rows > limit:
                rows -= 1
                break
            dst.write(line)
    return rows


def fetch_raw_file(spec: RawFileSpec, filename: str, output_dir: Path, limit: int | None) -> dict:
    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(
            repo_id=spec.hf_id,
            repo_type="dataset",
            filename=filename,
        )
    )
    out_path = output_dir / spec.key / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = None
    original_rows = None
    if filename.endswith(".jsonl"):
        original_rows = count_jsonl_lines(source)
        if limit is not None:
            rows = copy_limited_jsonl(source, out_path, limit)
        else:
            shutil.copyfile(source, out_path)
            rows = original_rows
    else:
        shutil.copyfile(source, out_path)

    return {
        "dataset": spec.key,
        "hf_id": spec.hf_id,
        "file": filename,
        "rows": rows,
        "original_rows": original_rows,
        "path": str(out_path.as_posix()),
        "note": spec.note,
    }


def fetch_snapshot(spec: SnapshotSpec, output_dir: Path) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    repo_files = HfApi().list_repo_files(spec.hf_id, repo_type="dataset")
    matched_files = []
    seen = set()
    for pattern in spec.allow_patterns:
        for path in repo_files:
            if path not in seen and fnmatch.fnmatch(path, pattern):
                matched_files.append(path)
                seen.add(path)

    entries = []
    for relative in matched_files:
        source = Path(
            hf_hub_download(
                repo_id=spec.hf_id,
                repo_type="dataset",
                filename=relative,
            )
        )
        target = output_dir / spec.key / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries.append(
            {
                "dataset": spec.key,
                "hf_id": spec.hf_id,
                "file": relative,
                "bytes": target.stat().st_size,
                "path": str(target.as_posix()),
                "note": spec.note,
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for spec in iter_specs(args.datasets):
        for split in spec.splits:
            print(f"Fetching {spec.key}:{split} from {spec.hf_id}")
            entries.append(fetch_split(spec, split, output_dir, args.limit))
    for spec in iter_raw_file_specs(args.datasets):
        for filename in spec.files:
            print(f"Fetching {spec.key}:{filename} from {spec.hf_id}")
            entries.append(fetch_raw_file(spec, filename, output_dir, args.limit))
    for spec in iter_snapshot_specs(args.datasets):
        print(f"Fetching {spec.key} snapshot from {spec.hf_id}")
        entries.extend(fetch_snapshot(spec, output_dir))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
