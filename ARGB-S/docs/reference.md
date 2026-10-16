# Technical reference

See [README.md](../README.md) for installation and basic usage, and the [configuration guide](configuration.md) for field definitions and control-variable relationships.

## Generation steps

1. Sample the requested layer count and entry width, and create entry nodes.
2. Sample each emitting node's out-degree and jump distances, recording edge slots for future layers.
3. Create each regular layer from its pending slot count, bounded by the width cap and remaining node budget.
4. Assign parents to cover every child, then assign remaining slots randomly, removing duplicate edges. Layers without pending slots use an earlier-layer parent as a fallback.
5. When enabled, create the reserved converge node at the requested final layer or earlier if the budget runs out.
6. Patch non-terminal nodes without successors to an existing next-layer node, then sum resource loads.

## JSON dataset format

Datasets have a top-level `graphs` array. Each entry contains a graph ID, aggregate resource loads, nodes, and edges; for example:

```json
{
  "id": "000001",
  "estimated_cost": {
    "total_tokens": 1,
    "total_cpu_time_ms": 1,
    "total_gpu_time_ms": 1
  },
  "nodes": [
    {"node_id": "N_0_0", "node_type": "GPU", "layer": 0, "tokens": {"tokens": 1}, "time_ms": 1},
    {"node_id": "N_1_0", "node_type": "CPU", "layer": 1, "time_ms": 1}
  ],
  "edges": [
    {"edge_id": "E0", "source": "N_0_0", "target": "N_1_0"}
  ]
}
```

Both CPU and GPU nodes carry `time_ms`; GPU nodes additionally carry `tokens`. Aggregate CPU and GPU times are sums of the respective node execution times and cannot be treated as a schedule's elapsed runtime.

`generate_and_save()` records the final merged generator configuration in the dataset's top-level `config` field. The standalone writers include it when passed `config=...`. The variant without layers omits only node `layer` fields; its recorded generator configuration is preserved.

## Public API

All entries below are imported from `graph_generator`.

| API | Purpose |
|---|---|
| `load_config(config_path=None, **overrides)` | Merge and validate generator settings |
| `get_config_path(name="default.toml")` | Locate a packaged TOML preset |
| `generate_graph(id="000001", config_path=None, seed=None, config=None, **overrides)` | Generate one `DirectedGraph` |
| `RunConfig(...)` | Set batch generation and export parameters; `generator` is required |
| `generate_graphs(config)` | Generate all graphs and retain them in a list |
| `iter_graphs(config)` | Yield graphs in index order, with bounded pending jobs |
| `generate_and_save(config, retain_graphs=True)` | Generate and stream the enabled JSON variants |
| `save_dataset(graphs, output_path, config=None)` | Write JSON with node layers |
| `save_dataset_no_layer(graphs, output_path, config=None)` | Write JSON without node layers |
| `visualize(input_path, output_path, graph_index=0, max_nodes_per_layer=500)` | Write a standalone HTML visualization |
| `graph_to_dict(graph)` / `graphs_to_dataset(graphs, config=None)` | Convert graphs into in-memory export objects |
| `config_to_dict(config)` | Convert the configuration to plain nested data |

Configuration types are `GeneratorConfig`, `TopologyConfig`, `ResourceConfig`, and `NormalIntDistribution`. Graph types are `Node`, `Edge`, and `DirectedGraph`; the graph exposes `nodes()`, `edges()`, `successors()`, `predecessors()`, and `nodes_in_layer()`.

### Batch defaults

| `RunConfig` field | Default |
|---|---|
| `graph_count` | `10` |
| `thread_count` | `4` |
| `base_seed` | `7` |
| `output_path_with_layer` | `./outputs/generated_graphs.json` |
| `output_path_without_layer` | `./outputs/generated_graphs_no_layer.json` |
| `save_with_layer` / `save_without_layer` | Both `True` |

At least one output variant must be enabled, and enabled variants must use distinct output paths. Batch size may be zero; worker count must be at least one. Relative output paths are resolved from the current working directory. The example script anchors its files to `outputs/example_outputs/` at the repository root.

### Streaming export

Writers accept one-pass iterators and serialize one graph at a time. `generate_and_save()` writes both variants in one traversal; by default it also retains and returns all generated graphs.

```python
from graph_generator import RunConfig, load_config, generate_and_save

run = RunConfig(graph_count=100, generator=load_config())
generate_and_save(run, retain_graphs=False)  # Writes datasets and returns an empty list.
```

`iter_graphs()` keeps at most `thread_count` pending jobs and yields graphs in index order. Memory still depends on each graph's size and the worker count. HTML visualization loads the dataset into memory and does not stream it. Generation or writing failures may leave incomplete JSON files.
