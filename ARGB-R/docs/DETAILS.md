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
  docs/
    DETAILS.md
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

root = Path("ARGB-R")
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

## Integrity and provenance

`manifest.json` records data version, source selection, graph statistics,
resource profiles, and file hashes. Its file paths are relative to the ARGB
repository root and begin with `ARGB-R/`. ToolBench index paths are relative to
the index directory.

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

Paths in this table are relative to `ARGB-R/strategyqa/`.

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

Paths in this table are relative to `ARGB-R/musique/`.

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

Paths in this table are relative to `ARGB-R/toolbench/`.

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

Paths in this table are relative to `ARGB-R/alfworld/`.

| File | Role |
|---|---|
| `dags.json` | 6,374 DAGs (76,455 nodes and 79,065 edges) |
| `resources.json` | Matching calibrated resource sidecar |

The earlier balanced 100-DAG sample is not part of this paper-aligned release.
