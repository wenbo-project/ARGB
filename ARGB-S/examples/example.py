"""Examples for generating, exporting, and visualizing graphs.

Demonstrates the three configuration styles supported by the design:

1. pure defaults (from ``graph_generator/configs/default.toml``);
2. packaged TOML preset (e.g. ``graph_generator/configs/large_graph.toml``);
3. TOML + temporary parameter overrides (highest precedence).
"""

import sys
from pathlib import Path

# Allow running this file directly from the repository without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_generator import (
    RunConfig,
    generate_graphs,
    generate_graph,
    graphs_to_dataset,
    load_config,
    get_config_path,
    save_dataset,
    visualize,
    generate_and_save,
)

# Every artifact produced by this script lands here. The path is anchored to
# this file (not to the current working directory), so running the example from
# anywhere always writes to ``outputs/example_outputs/`` at the project root.
output_dir = Path(__file__).resolve().parents[1] / "outputs" / "example_outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# 1) Pure defaults.
default_config = load_config()

# 2) Packaged TOML override.
large_config = load_config(get_config_path("large_graph.toml"))

# 3) TOML + temporary overrides (highest precedence).
tree_config = load_config(
    get_config_path("tree.toml"), out_degree={"mean": 2}
)


# Generate a small batch, inspect it in memory, save it as JSON, and render a
# standalone HTML visualization.
run_config = RunConfig(
    graph_count=3,
    thread_count=1,
    base_seed=42,
    generator=default_config,
)

graphs = generate_graphs(run_config)
dataset = graphs_to_dataset(graphs, config=run_config.generator)

json_path = output_dir / "example_graphs.json"
html_path = output_dir / "example_graphs.html"

save_dataset(graphs, json_path, config=run_config.generator)
visualize(input_path=json_path, output_path=html_path)

print("\nfiles written")
print(f"  JSON: {json_path}")
print(f"  HTML: {html_path}")


# Single graph with a user TOML plus one temporary override.
single = generate_graph(
    id="custom",
    config_path=get_config_path("large_graph.toml"),
    out_degree={"mean": 4},
    seed=7,
)
print(
    f"\nsingle graph: {single.graph['id']} "
    f"nodes={len(single.nodes())} edges={len(single.edges())}"
)


# Simplified default run: 10 graphs with the default generator settings, written
# to outputs/example_outputs/generated_graphs.json and generated_graphs_no_layer.json.
# The JSON top level records the final merged generator config.
default_run = RunConfig(
    generator=default_config,
    output_path_with_layer=str(output_dir / "generated_graphs.json"),
    output_path_without_layer=str(output_dir / "generated_graphs_no_layer.json"),
)
default_graphs = generate_and_save(default_run)
default_html_path = output_dir / "graph_view.html"
visualize(
    input_path=default_run.output_path_with_layer,
    output_path=default_html_path,
)

print("\ndefault config files written")
print(f"  JSON: {default_run.output_path_with_layer}")
print(f"  HTML: {default_html_path}")
