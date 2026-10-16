"""Dataset conversion, generation, and persistence helpers."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import GeneratorConfig, RunConfig
from .generator import generate_graph
from .graph_structure import Edge, DirectedGraph, Node


def generate_graphs(config: RunConfig) -> list[DirectedGraph]:
    """Generate all graphs for a batch run, retaining them in memory."""
    return list(iter_graphs(config))


def iter_graphs(config: RunConfig) -> Iterator[DirectedGraph]:
    """Yield graphs in index order with at most thread_count pending jobs."""
    config.validate()
    worker_count = max(1, min(config.thread_count, config.graph_count or 1))

    def generate_one_graph(index: int) -> DirectedGraph:
        seed = None if config.base_seed is None else config.base_seed * 1000 + index
        return generate_graph(
            id=f"{index:06d}",
            seed=seed,
            config=config.generator,
        )

    if worker_count == 1:
        for index in range(config.graph_count):
            yield generate_one_graph(index)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        indices = iter(range(config.graph_count))
        pending = deque(
            executor.submit(generate_one_graph, index)
            for index in range(worker_count)
        )
        for _ in range(worker_count):
            next(indices)
        while pending:
            yield pending.popleft().result()
            index = next(indices, None)
            if index is not None:
                pending.append(executor.submit(generate_one_graph, index))


def generate_and_save(
    config: RunConfig, *, retain_graphs: bool = True,
) -> list[DirectedGraph]:
    """Generate and stream graphs into the configured JSON arrays.

    The JSON output records the final, merged generator config so every
    experiment stays reproducible and traceable. Set retain_graphs=False for
    bounded-memory generation; in that mode the return value is an empty list.
    """
    config.validate()
    outputs: list[tuple[str | Path, bool]] = []
    if config.save_with_layer:
        outputs.append((config.output_path_with_layer, True))
    if config.save_without_layer:
        outputs.append((config.output_path_without_layer, False))

    graphs: list[DirectedGraph] = []

    def generated() -> Iterator[DirectedGraph]:
        for graph in iter_graphs(config):
            if retain_graphs:
                graphs.append(graph)
            yield graph

    _write_datasets(generated(), outputs, config=config.generator)
    return graphs


def config_to_dict(config: GeneratorConfig) -> dict[str, Any]:
    """Serialize a GeneratorConfig (or any nested dataclass) to plain data."""
    return _to_plain(config)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _to_plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def graph_to_dict(graph: DirectedGraph) -> dict:
    """Convert a DirectedGraph instance to the exported JSON shape."""
    return {
        "id": graph.graph["id"],
        "estimated_cost": graph.graph.get("estimated_cost", {}),
        "nodes": [_node_to_dict(node) for node in graph.nodes()],
        "edges": [_edge_to_dict(edge) for edge in graph.edges()],
    }


def graphs_to_dataset(
    graphs: Iterable[DirectedGraph],
    config: GeneratorConfig | None = None,
) -> dict:
    """Wrap generated graphs in the top-level dataset object."""
    data: dict[str, Any] = {
        "graphs": [graph_to_dict(graph) for graph in graphs]
    }
    if config is not None:
        data["config"] = config_to_dict(config)
    return data


def save_dataset(
    graphs: Iterable[DirectedGraph],
    output_path: str | Path,
    config: GeneratorConfig | None = None,
) -> None:
    """Stream generated graphs into a UTF-8 JSON array."""
    _write_datasets(graphs, [(output_path, True)], config=config)


def save_dataset_no_layer(
    graphs: Iterable[DirectedGraph],
    output_path: str | Path,
    config: GeneratorConfig | None = None,
) -> None:
    """Stream generated graphs into a JSON array without layer fields."""
    _write_datasets(graphs, [(output_path, False)], config=config)


def _write_datasets(
    graphs: Iterable[DirectedGraph],
    outputs: list[tuple[str | Path, bool]],
    config: GeneratorConfig | None = None,
) -> None:
    """Consume graphs once and serialize one graph at a time to each output."""
    paths = [(Path(path), with_layer) for path, with_layer in outputs]
    if len({path.resolve() for path, _ in paths}) != len(paths):
        raise ValueError("output paths must be distinct")
    # Write with-layer outputs first so one graph dictionary serves both variants.
    paths.sort(key=lambda item: not item[1])
    with ExitStack() as stack:
        writers = []
        for path, with_layer in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = stack.enter_context(path.open("w", encoding="utf-8"))
            writer.write('{\n  "graphs": [')
            writers.append((writer, with_layer))

        first = True
        for graph in graphs:
            graph_data = graph_to_dict(graph)
            for writer, with_layer in writers:
                if not with_layer:
                    for node in graph_data["nodes"]:
                        node.pop("layer", None)
                writer.write("\n" if first else ",\n")
                json.dump(graph_data, writer, ensure_ascii=False, indent=2)
            first = False

        for writer, _ in writers:
            writer.write("\n  ]")
            if config is not None:
                writer.write(',\n  "config": ')
                json.dump(config_to_dict(config), writer, ensure_ascii=False, indent=2)
            writer.write("\n}\n")


def _node_to_dict(node: Node) -> dict:
    data = {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "layer": node.layer,
    }
    if node.tokens:
        data["tokens"] = node.tokens
    if node.time_ms is not None:
        data["time_ms"] = node.time_ms
    return data


def _edge_to_dict(edge: Edge) -> dict:
    return {
        "edge_id": edge.edge_id,
        "source": edge.source,
        "target": edge.target,
    }
