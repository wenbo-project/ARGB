# ARGB-R Dataset Details

See the [ARGB README](../../README.md) for the benchmark overview and ARGB-S usage.

ARGB-R contains reasoning and execution graphs derived from StrategyQA, MuSiQue,
ToolBench/ToolLLM, and ALFWorld. The four sources cover factual reasoning,
multi-hop evidence reasoning, tool use, and embodied interaction.

Data release version: **1.1.0**. This package preserves the frozen graph and
resource files byte for byte. No model conversion or resource estimation was
rerun during packaging.

## Files

```text
ARGB-R/
  datasets/
    manifest.json
    strategyqa/
      dags.json
      resources.json
      README.md
    musique/
      dags.json
      resources.json
      README.md
    toolbench/
      dags-00001-of-00002.json
      dags-00002-of-00002.json
      dags.index.json
      resources.json
      README.md
    alfworld/
      dags.json
      resources.json
      README.md
  docs/
    DETAILS.md
  script/
    convert_dag.py
    src/
```

## Inventory

| Dataset | Graphs | Nodes | Edges | GPU nodes | CPU nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| StrategyQA | 2,061 | 7,642 | 6,090 | 5,623 | 2,019 |
| MuSiQue | 19,938 | 95,044 | 85,668 | 54,497 | 40,547 |
| ToolBench | 26,999 | 124,408 | 131,936 | 51,136 | 73,272 |
| ALFWorld | 6,374 | 76,455 | 79,065 | 33,908 | 42,547 |
| Total | 55,372 | 303,549 | 302,759 | 145,164 | 158,385 |

`source_split: train` in the manifest identifies the upstream records used for
conversion; this package does not introduce a new training/test split.

## Loading the data

Each `dags.json` is a JSON array. Graphs retain their released field names,
including `dag_id`, `nodes`, and `edges`. Nodes include `node_id`, `node_type`,
and an execution `prompt`; edges identify `source` and `target` nodes. This
package preserves the existing ARGB-R format without converting it to the
ARGB-S export schema.

```python
import json
from pathlib import Path

root = Path("ARGB-R/datasets")
with (root / "strategyqa" / "dags.json").open(encoding="utf-8") as stream:
    graphs = json.load(stream)
with (root / "strategyqa" / "resources.json").open(encoding="utf-8") as stream:
    resources = json.load(stream)
```

Run this example from the ARGB repository root. For ToolBench, read
`dags.index.json`, resolve each shard path relative to the index directory, and
load the two JSON arrays in listed order. Together they contain 26,999 graphs
and use one shared `resources.json`.

Keep each graph file with its matching `resources.json`. Resource files use
the existing `wenbo_resource_estimates_v1` schema and `local_sim_v2` profile.
CPU durations represent simulated local occupancy, rather than external
service latency. Historical schema names and source paths inside data metadata
are retained as provenance; current file locations are given by the manifest.

## Conversion tools

Use [convert_dag.py](../script/convert_dag.py) to convert the four source datasets
into graph and resource JSON files. Supporting code is in `ARGB-R/script/src/`,
including dataset-specific prompts and conversion rules, model requests, JSON
parsing, format normalization, and resource estimation.

### Requirements and layout

Conversion, CPU calibration, and release verification require **Python 3.10 or
later** and use the standard library. Raw-data downloads additionally require
`datasets` and `huggingface-hub`; running the tests requires `pytest`.

```text
script/
  convert_dag.py
  src/
    __init__.py
    _strategyqa.py
    _musique.py
    _toolbench.py
    _alfworld.py
    dag_conversion_common.py
    dag_standard.py
    fetch_raw_datasets.py
    calibrate_cpu_resource_estimates.py
    verify_data.py
    tests/
```

From the ARGB repository root:

```sh
cd ARGB-R/script
python convert_dag.py --help
```

Run the following commands from `ARGB-R/script/`. Default input paths are anchored
to `ARGB-R/raw/`, and default output paths to `ARGB-R/generated/`, regardless of
the working directory. Explicit `--input`, `--input-root`, and `--output` paths
are resolved relative to the working directory when they are not absolute.

### Prepare the source data

| Dataset | Default input relative to `ARGB-R/` | Input structure |
| --- | --- | --- |
| StrategyQA | `raw/strategyqa/train.jsonl` | One question-and-facts record per line |
| MuSiQue | `raw/musique/train.jsonl` | One question-and-evidence record per line |
| ToolBench | `raw/toolbench/toolllama_G123_dfs_train.json` | A JSON array of tool-use conversations |
| ALFWorld | `raw/alfworld/json_2.1.1/` | Split directories containing task-level `traj_data.json` files |

To download the three supported sources:

```sh
python -m pip install "datasets>=2.18" "huggingface-hub>=0.23"
python -m src.fetch_raw_datasets --datasets strategyqa musique toolbench
```

Obtain ALFWorld trajectories separately from the official release and place the
`json_2.1.1` directory at the path above, or supply `--input-root` explicitly.

### Convert with an LLM

Set the API key in the shell. For PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
```

For Bash:

```sh
export DEEPSEEK_API_KEY="your-api-key"
```

The default converter is `llm`, with model `deepseek-v4-flash`, temperature `0.1`,
and base URL `https://api.deepseek.com`. The CLI also accepts `--model` and
`--llm-base-url`; `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` provide
compatible environment settings. LLM conversion calls the configured API with
each task's extracted information.

```sh
python convert_dag.py --dataset strategyqa --converter llm --limit 10
python convert_dag.py --dataset musique --converter llm --limit 10
python convert_dag.py --dataset toolbench --converter llm --limit 10
python convert_dag.py --dataset alfworld --converter llm --splits train --limit 10
```

For custom paths and concurrent requests:

```sh
python convert_dag.py --dataset musique --input ../raw/musique/train.jsonl --output ../generated/musique/dags.json --converter llm --limit 10 --workers 2 --checkpoint-every 5
```

`--limit` defaults to 100; `--limit 0` processes the available records from the
zero-based `--start` offset. `--workers` defaults to 1, and `--llm-retries` defaults
to 2 retries after the initial attempt. The default ALFWorld split is `train`.
Invalid ToolBench and ALFWorld LLM results are skipped after retries by default;
use `--no-skip-invalid-llm` to stop on such failures. Existing-output resume and
checkpoint behavior follow the individual backends; StrategyQA, MuSiQue, and
ALFWorld resume completed records, while ToolBench writes checkpoints but starts
the requested conversion again when rerun.

### Run an offline conversion

Use the rule converter to check the input/output pipeline without model requests:

```sh
python convert_dag.py --dataset strategyqa --converter rule --limit 10 --output ../generated/strategyqa-rule/dags.json
```

All four datasets support `--converter rule`. Rule-based outputs use predefined
graph construction rules; the released ARGB-R graphs use the frozen LLM
conversions described in the dataset-specific sections below. LLM-to-rule
fallback is disabled by default and can be enabled with `--allow-rule-fallback`.
`--output` selects the output file. The example saves rule-based graphs to
`ARGB-R/generated/strategyqa-rule/dags.json`.

### Outputs and resource calibration

Each conversion writes two files under `ARGB-R/generated/<dataset>/` by default:

- `dags.json`: a JSON array of standardized graphs, including execution prompts
  and CPU/GPU node labels.
- `dags_resource_estimates.json`: initial token, CPU, and edge-data estimates
  produced by the shared resource helper.

To produce a CPU resource sidecar with the release's `local_sim_v2` profile:

```sh
python -m src.calibrate_cpu_resource_estimates ../generated/strategyqa/dags.json
```

Calibration writes `dags_resource_estimates_local_sim_v2.json` beside the graph
file. Supply multiple graph paths to calibrate several files in one command.
`--force` replaces an existing calibrated file. `--include-node-context` adds
prompts and node metadata to the sidecar.

New conversions are saved under `ARGB-R/generated/`; release 1.1.0 is stored
under `ARGB-R/datasets/`. ToolBench conversion produces a single graph file.
Release sharding and manifest creation are separate steps. Each LLM run can
produce different graphs and resource estimates.

### Verification and tests

Verify the published files against `ARGB-R/datasets/manifest.json`:

```sh
python -m src.verify_data
```

The verifier checks graph and resource SHA-256 hashes, ToolBench shard hashes
and counts, graph identifiers and dependencies, resource pairing, and release
statistics. `--data-root` selects another release copy using its `ARGB-R` or
`ARGB-R/datasets` directory. Historical manifest paths are supported in both layouts.

Run the conversion package tests:

```sh
python -m pip install pytest
python -m pytest -q src/tests
```

The tests use small local samples and a local HTTP server to check all four
conversion backends, path resolution, model response parsing, retries, resource
calibration, and release validation.

## Integrity and provenance

`datasets/manifest.json` records data version, source selection, graph statistics,
resource profiles, and file hashes. It retains historical paths beginning with
`ARGB-R/`; for the current layout, resolve the remainder relative to
`ARGB-R/datasets/`. The verification tool handles this mapping without modifying
the manifest. ToolBench index paths remain relative to the index directory.

The `generated/` workspace, temporary conversions, and debug outputs are
excluded. Dataset-specific sources, conversions, and released files are
documented below: [StrategyQA](#strategyqa), [MuSiQue](#musique),
[ToolBench](#toolbench), and [ALFWorld](#alfworld).
Upstream attribution, licensing terms, and citation requirements remain
applicable to the corresponding source data.

## Dataset-specific details

### StrategyQA

#### Source and conversion

- Upstream source: StrategyQA training examples.
- Conversion: frozen full LLM rerun.
- Workload shape: yes/no multi-hop reasoning DAGs, commonly with parallel
  evidence or comparison branches followed by answer aggregation.

#### Released files

Paths in this table are relative to `ARGB-R/datasets/strategyqa/`.

| File | Role |
|---|---|
| `dags.json` | 2,061 DAGs (7,642 nodes and 6,090 edges) |
| `resources.json` | Matching calibrated resource sidecar |

### MuSiQue

#### Source and conversion

- Upstream source: MuSiQue `train.jsonl`.
- Conversion: frozen full LLM conversion using the optimized prompt and rerun
  configuration.
- Workload shape: multi-hop question-answering DAGs with decomposition,
  retrieval/reasoning branches, and final aggregation.

#### Released files

Paths in this table are relative to `ARGB-R/datasets/musique/`.

| File | Role |
|---|---|
| `dags.json` | 19,938 DAGs (95,044 nodes and 85,668 edges) |
| `resources.json` | Matching calibrated resource sidecar |

### ToolBench

#### Source and conversion

- Upstream source: ToolBench `toolllama_G123_dfs_train.json`.
- Conversion: the frozen merged 26,999-graph DeepSeek LLM conversion used by
  the paper's benchmark table and real-data experiments.
- Workload shape: agent tool-use workflows; independent tool calls may appear
  as parallel branches.

#### Released files

Paths in this table are relative to `ARGB-R/datasets/toolbench/`.

| File | Role |
| --- | --- |
| `dags-00001-of-00002.json` | First 13,500 graphs in original order |
| `dags-00002-of-00002.json` | Remaining 13,499 graphs |
| `dags.index.json` | Ordered shard paths, counts, and SHA-256 hashes |
| `resources.json` | Matching calibrated resource sidecar for both shards |

The two JSON arrays contain 26,999 graphs, 124,408 nodes, and 131,936 edges.
Read the index, resolve shard paths relative to its directory, and load the
arrays in the listed order. Keep the shared resource file with both shards.

Graph IDs, structure, node order, resource labels, and frozen numerical costs
are preserved. Historical schema identifiers and provenance metadata remain
unchanged. Partial checkpoints, the earlier 3,000-graph subset, and API/debug
samples are excluded.

### ALFWorld

#### Source and conversion

- Upstream source: ALFWorld `json_2.1.1/train` trajectories.
- Conversion: OpenAI-compatible LLM conversion with DeepSeek.
- Selection: the frozen 6,374-DAG full LLM conversion used by the paper's
  benchmark table and real-data experiments.

#### Released files

Paths in this table are relative to `ARGB-R/datasets/alfworld/`.

| File | Role |
|---|---|
| `dags.json` | 6,374 DAGs (76,455 nodes and 79,065 edges) |
| `resources.json` | Matching calibrated resource sidecar |

The earlier balanced 100-DAG sample is not part of this paper-aligned release.
