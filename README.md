# ARGB: Agent Reasoning Graph Benchmark

**Graph datasets for heterogeneous agent reasoning and CPU/GPU scheduling.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](ARGB-S/pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Read-blue?style=flat-square)](#documentation)

The Agent Reasoning Graph Benchmark (ARGB) combines graphs derived from public agent reasoning tasks (**ARGB-R**) with configurable synthetic workloads (**ARGB-S**) for controlled experiments and evaluation across task domains.

| Component | Data source | Primary use |
| --- | --- | --- |
| [**ARGB-R**](#argb-r-real-task-derived-dataset) | 55,372 graphs derived from StrategyQA, MuSiQue, ToolBench/ToolLLM, and ALFWorld | Evaluation on workloads derived from real tasks |
| [**ARGB-S**](#argb-s-synthetic-dataset) | Synthetic graphs with configurable structures and resource loads | Controlled experiments on graph structure and CPU/GPU resource composition |

## Repository Structure

```text
ARGB/
├── ARGB-R/      # Real-task-derived graphs, resource files, and dataset documentation
├── ARGB-S/      # Synthetic graph generator, configurations, examples, and tests
├── README.md
└── LICENSE
```

## ARGB-R: Real-Task-Derived Dataset

ARGB-R is derived from four public agent reasoning task datasets, covering factual reasoning, multi-hop evidence reasoning, tool use, and embodied interaction.

### Source Datasets

| Dataset | Task domain | Information extracted |
|---|---|---|
| [StrategyQA](https://github.com/eladsegal/strategyqa) | Factual reasoning | Questions and supporting facts |
| [MuSiQue](https://github.com/stonybrooknlp/musique) | Multi-hop evidence reasoning | Questions and evidence passages |
| [ToolBench / ToolLLM](https://github.com/OpenBMB/ToolBench) | Tool use | Task descriptions, available tools, and invocation traces |
| [ALFWorld](https://github.com/alfworld/alfworld) | Embodied interaction | Goals, environment descriptions, and action sequences |

### Construction Pipeline

Reasoning graphs are not inherently subject to an acyclicity constraint. To simplify the experimental setup, we generate executable directed acyclic graphs (DAGs) as experimental instances.

1. **Extract task information.** Collect the dataset-specific information listed above and retain the source dataset for each task.
2. **Construct dependency graphs.** Provide the extracted information to `deepseek-v4-flash` and prompt it to construct a DAG representing the task's reasoning and execution dependencies. Subtasks become operation nodes, and directed edges encode prerequisite relationships.
3. **Normalize the graphs.** Convert the generated graphs into a unified JSON representation for consistent workload processing and scheduling across task domains.

### Resource Assignment and Representation

| Operation category | Resource label |
|---|---|
| LLM-intensive reasoning, planning, and answer generation | GPU |
| Tool invocations, environment interactions, and deterministic processing | CPU |

Each normalized graph records its source dataset. Each node contains an identifier, an execution prompt, and a CPU/GPU resource label. Each edge specifies its source and destination node identifiers.

The released graphs and matching resource files are organized under `ARGB-R/`. See the [ARGB-R dataset details](ARGB-R/docs/DETAILS.md) for dataset-specific sources and conversions, release statistics, loading instructions, resource semantics, and file integrity information. Read ToolBench shards in the order listed in `dags.index.json` and keep each graph file with its matching `resources.json`.

## ARGB-S: Synthetic Dataset

ARGB-S generates synthetic layered dependency graphs using the `graph_generator` Python package. Nodes represent CPU or GPU operations with sampled execution times; GPU nodes also carry token counts. Directed edges represent dependencies between operations. TOML configuration controls graph shape, CPU/GPU sampling probabilities, and resource load distributions.

### Quick Start

Requires **Python 3.10 or later**. From the repository root:

```sh
cd ARGB-S
python -m pip install -e .
python examples/example.py
```

The example writes JSON and HTML files to `ARGB-S/outputs/example_outputs/`. Open `ARGB-S/outputs/example_outputs/example_graphs.html` to inspect the generated graphs. Paths in the following Python examples are relative to `ARGB-S/` when it is the working directory.

### Generate a Single Graph

```python
from graph_generator import generate_graph

graph = generate_graph(id="000001", seed=1)
print(len(graph.nodes()), len(graph.edges()))
```

This uses the packaged default configuration. Reuse the same configuration and random seed to reproduce the generated graph.

### Generate a Batch, Export, and Visualize

```python
from graph_generator import (
    RunConfig,
    load_config,
    visualize,
    generate_and_save,
)

run = RunConfig(
    graph_count=3,
    thread_count=1,
    base_seed=42,
    generator=load_config(),
)
graphs = generate_and_save(run)
visualize(
    input_path=run.output_path_with_layer,
    output_path="outputs/graph_view.html",
)
```

`load_config()` loads the default generator configuration. TOML files contain `[generator]` settings; batch size, worker count, base seed, and output paths are set in `RunConfig`. When run from `ARGB-S/`, this batch configuration produces:

- `ARGB-S/outputs/generated_graphs.json`: a graph dataset with node layer information and the merged generator configuration.
- `ARGB-S/outputs/generated_graphs_no_layer.json`: the same dataset with node `layer` fields omitted.
- `ARGB-S/outputs/graph_view.html`: a standalone HTML visualization.

For large batches, use `generate_and_save(run, retain_graphs=False)` to stream the JSON output.

Packaged presets, custom TOML files, parameter overrides, and the `gpu_ratio` setting are covered in the [ARGB-S configuration guide](ARGB-S/docs/configuration.md).

## Documentation

- [ARGB-R dataset details](ARGB-R/docs/DETAILS.md): source-specific conversion details, release inventory, data loading, resource semantics, and integrity information.
- [ARGB-S configuration guide](ARGB-S/docs/configuration.md): presets, all parameters, sampling rules, and how graph shape controls interact.
- [ARGB-S technical reference](ARGB-S/docs/reference.md): generation steps, JSON format, public API, batch defaults, and streaming export.

## License

The code in this repository is released under the [MIT License](LICENSE). Upstream attribution, licensing terms, and citation requirements for the source datasets are covered in the [ARGB-R dataset details](ARGB-R/docs/DETAILS.md#integrity-and-provenance).
