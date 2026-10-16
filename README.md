# graph_generator

A configurable Python library for generating **synthetic layered dependency graphs** for task dependency and scheduling experiments. Nodes represent CPU or GPU tasks with sampled execution times; GPU nodes also carry token counts. Edges represent task dependencies.Reasoning graphs are not inherently subject to an acyclicity constraint. To simplify the experimental setup, we generate executable directed acyclic graphs (DAGs) as experimental instances.

Generate one graph or a batch, export JSON datasets, and inspect the graphs in a standalone HTML visualization. Graph shape and resource load distributions are configured through TOML files.

## Quick start

Requires **Python 3.10 or later**. From the repository root:

```sh
python -m pip install -e .
python examples/example.py
```

The example writes JSON and HTML files to `outputs/example_outputs/`. Open `outputs/example_outputs/example_graphs.html` to inspect the generated graphs. Run the example script to try the complete generation and export workflow.

## Basic usage

### Generate a single graph

```python
from graph_generator import generate_graph

graph = generate_graph(id="000001", seed=1)
print(len(graph.nodes()), len(graph.edges()))
```

This uses the packaged default configuration. Reuse the same configuration and random seed to reproduce the generated graph.

### Choose a preset and adjust parameters

```python
from graph_generator import generate_graph, get_config_path

graph = generate_graph(
    id="custom",
    config_path=get_config_path("large_graph.toml"),
    out_degree={"mean": 4},
    seed=17,
)
```

`get_config_path("large_graph.toml")` returns the absolute path of the packaged `graph_generator/configs/large_graph.toml`, independently of the current working directory. This example first loads `default.toml`, then merges the preset, and finally overrides the out-degree mean with `out_degree={"mean": 4}`. Fields that are not overridden retain their existing values. To use your own TOML file, pass its path directly, for example `config_path="my_config.toml"`.

| Preset | Use |
|---|---|
| [default.toml](graph_generator/configs/default.toml) | Default settings: 40 requested layers, adjacent-layer edges, one final converge node |
| [large_graph.toml](graph_generator/configs/large_graph.toml) | Higher out-degree mean and edges spanning up to three layers |
| [tree.toml](graph_generator/configs/tree.toml) | Five-layer tree with three children per non-terminal node and no forced convergence |

### Generate a batch, export, and visualize

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

`load_config()` loads the default generator configuration. TOML files contain `[generator]` settings; batch size, worker count, base seed, and output paths are set in `RunConfig`. Running the example produces:

- `outputs/generated_graphs.json`: a graph dataset with node layer information and the merged generator configuration.
- `outputs/generated_graphs_no_layer.json`: the same dataset with node `layer` fields omitted.
- `outputs/graph_view.html`: a standalone HTML visualization.

For large batches, use `generate_and_save(run, retain_graphs=False)` to stream the JSON output.

## Further documentation

- [Configuration guide](docs/configuration.md): all parameters, sampling rules, and how out-degree, layer jumps, width limits, node budget, and convergence interact.
- [Technical reference](docs/reference.md): generation steps, JSON format, public API, batch defaults, and streaming export.
