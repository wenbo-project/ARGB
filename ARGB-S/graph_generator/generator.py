"""Layered synthetic graph generation.

The graph is grown top-down from the entry layer:

- every parent node emits ``out_degree`` edge slots;
- each slot records the parent into ``pending_edges`` for a future layer
  (this is how edges can jump across multiple layers);
- when a layer is created, its width is derived from the number of pending
  slots, capped by the layer width limit and remaining node budget;
- edges are assigned "cover first, then random" so every child has in-degree
  at least one and the same parent never connects twice to the same child.
- non-terminal nodes without successors connect to a random next-layer node
  after all layer widths have been fixed.

Scale protection comes from ``max_layer_width`` and ``max_total_nodes``;
``force_converge`` reserves one node within the total budget for convergence.
"""

from __future__ import annotations

from collections import defaultdict
import random

from .config import (
    GeneratorConfig,
    TopologyConfig,
    load_config,
)
from .graph_structure import Edge, DirectedGraph, Node


def generate_graph(
    id: str = "000001",
    config_path: str | None = None,
    seed: int | None = None,
    config: GeneratorConfig | None = None,
    **overrides,
) -> DirectedGraph:
    """Generate a layered graph from either a config object or TOML/overrides."""
    if config is not None and (config_path is not None or overrides):
        raise ValueError(
            "config cannot be combined with config_path or overrides; "
            "build the final config with load_config() first"
        )
    if config is None:
        config = load_config(config_path, **overrides)
    else:
        config.validate()

    rng = random.Random(seed)

    layer_count = _sample_normal_int(
        rng=rng,
        mean=config.topology.layer_count.mean,
        variance=config.topology.layer_count.variance,
        minimum=config.topology.layer_count.minimum,
        maximum=config.topology.layer_count.maximum,
    )
    config.validate_layer_count(layer_count)

    topo = config.topology
    graph = DirectedGraph(id=id)
    pending_edges: dict[int, list[str]] = defaultdict(list)

    entry_nodes = _create_entry_layer(graph=graph, config=config, rng=rng)
    _emit_out_edges(
        graph=graph,
        parents=entry_nodes,
        source_layer=0,
        layer_count=layer_count,
        pending_edges=pending_edges,
        rng=rng,
        config=config,
    )

    total_nodes = len(entry_nodes)
    regular_node_budget = topo.max_total_nodes - int(topo.force_converge)

    for child_layer in range(1, layer_count):
        is_last = child_layer == layer_count - 1

        # The final converge node uses its reserved slot, even if the regular
        # node budget has been exhausted on the preceding layer.
        if is_last and topo.force_converge:
            _converge(
                graph=graph,
                last_layer=child_layer,
                pending_edges=pending_edges,
                rng=rng,
                config=config,
            )
            break

        remaining_budget = regular_node_budget - total_nodes
        if remaining_budget < 1:
            break

        slots = pending_edges.pop(child_layer, [])
        width = _compute_layer_width(
            slot_count=len(slots),
            max_layer_width=topo.max_layer_width,
        )
        width = min(width, remaining_budget)

        children = _create_layer_nodes(
            graph=graph,
            layer=child_layer,
            count=width,
            config=config,
            rng=rng,
        )
        total_nodes += len(children)

        _assign_parents(
            graph=graph,
            slots=slots,
            children=children,
            child_layer=child_layer,
            rng=rng,
            config=config,
        )
        _emit_out_edges(
            graph=graph,
            parents=children,
            source_layer=child_layer,
            layer_count=layer_count,
            pending_edges=pending_edges,
            rng=rng,
            config=config,
        )

    if topo.force_converge and _max_layer(graph) < layer_count - 1:
        _converge_emergency(graph=graph, rng=rng, config=config)

    _ensure_successors(graph, rng)
    graph.graph["estimated_cost"] = _estimate_cost(graph)
    return graph


# ---------------------------------------------------------------------------
# Layer construction
# ---------------------------------------------------------------------------


def _create_entry_layer(
    graph: DirectedGraph,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Node]:
    """Sample the entry width once per graph, then create the entry layer (L0)."""
    distribution = config.topology.first_layer_width
    width = _sample_normal_int(
        rng=rng,
        mean=distribution.mean,
        variance=distribution.variance,
        minimum=distribution.minimum,
        maximum=distribution.maximum,
    )
    if config.topology.max_total_nodes < width + 1:
        raise ValueError(
            "topology.max_total_nodes must be at least sampled first_layer_width + 1 "
            f"({width + 1}): {config.topology.max_total_nodes}"
        )
    nodes = []
    for i in range(width):
        node = _make_node(
            node_id=f"N_0_{i}",
            layer=0,
            rng=rng,
            config=config,
        )
        graph.add_node(node)
        nodes.append(node)
    return nodes


def _create_layer_nodes(
    graph: DirectedGraph,
    layer: int,
    count: int,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Node]:
    """Create ``count`` nodes in one layer and return them."""
    nodes = []
    start_index = len(graph.nodes_in_layer(layer))
    for i in range(count):
        node = _make_node(
            node_id=f"N_{layer}_{start_index + i}",
            layer=layer,
            rng=rng,
            config=config,
        )
        graph.add_node(node)
        nodes.append(node)
    return nodes


def _converge(
    graph: DirectedGraph,
    last_layer: int,
    pending_edges: dict[int, list[str]],
    rng: random.Random,
    config: GeneratorConfig,
) -> Node:
    """Create a single converge node on the last layer and route parents into it."""
    converge = _create_layer_nodes(
        graph=graph, layer=last_layer, count=1, config=config, rng=rng
    )[0]
    candidates = pending_edges.pop(last_layer, [])
    for node in graph.nodes_in_layer(last_layer - 1):
        _add_edge(graph, node.node_id, converge.node_id)
    for parent in candidates:
        _add_edge(graph, parent, converge.node_id)
    return converge


def _converge_emergency(
    graph: DirectedGraph,
    rng: random.Random,
    config: GeneratorConfig,
) -> Node:
    """Use the reserved slot when regular nodes exhaust their budget early."""
    last_layer = _max_layer(graph)
    converge_layer = last_layer + 1
    converge = _make_node(
        node_id=f"N_{converge_layer}_0",
        layer=converge_layer,
        rng=rng,
        config=config,
    )
    graph.add_node(converge)
    for node in graph.nodes_in_layer(last_layer):
        _add_edge(graph, node.node_id, converge.node_id)
    return converge


def _max_layer(graph: DirectedGraph) -> int:
    return max(node.layer for node in graph.nodes())


def _compute_layer_width(
    slot_count: int,
    max_layer_width: int,
) -> int:
    """Use one child per pending slot, capped by the layer width limit."""
    return max(1, min(slot_count, max_layer_width))


# ---------------------------------------------------------------------------
# Edge management
# ---------------------------------------------------------------------------


def _assign_parents(
    graph: DirectedGraph,
    slots: list[str],
    children: list[Node],
    child_layer: int,
    rng: random.Random,
    config: GeneratorConfig,
) -> None:
    """Assign pending parent slots: cover first, then random.

    - Round 1 guarantees every child at least one parent (in-degree >= 1).
    - If slots run out, fall back to earlier-layer nodes to keep the graph
      connected.
    - Round 2 distributes the remaining slots randomly; ``_add_edge``
      deduplicates so one parent never connects to the same child twice.
    """
    if not children:
        return

    rng.shuffle(slots)

    for child in children:
        if not slots:
            break
        _add_edge(graph, slots.pop(), child.node_id)

    leftover = []
    for child in children:
        if graph.in_degree(child.node_id) == 0:
            leftover.append(child)
    if leftover:
        fallback = _earlier_layer_nodes(
            graph, child_layer, config.topology.max_layer_jump
        )
        if fallback:
            for child in leftover:
                _add_edge(graph, rng.choice(fallback).node_id, child.node_id)

    for parent in slots:
        _add_edge(graph, parent, rng.choice(children).node_id)


def _earlier_layer_nodes(
    graph: DirectedGraph,
    child_layer: int,
    max_layer_jump: int,
) -> list[Node]:
    """Return earlier-layer nodes eligible for fallback/extra dependencies."""
    nodes: list[Node] = []
    for layer in range(max(0, child_layer - max_layer_jump), child_layer):
        nodes.extend(graph.nodes_in_layer(layer))
    return nodes


def _ensure_successors(graph: DirectedGraph, rng: random.Random) -> None:
    # Add dependencies without creating nodes or changing layer widths.
    for layer in range(_max_layer(graph)):
        children = graph.nodes_in_layer(layer + 1)
        for parent in graph.nodes_in_layer(layer):
            if graph.out_degree(parent.node_id) == 0:
                _add_edge(graph, parent.node_id, rng.choice(children).node_id)


def _emit_out_edges(
    graph: DirectedGraph,
    parents: list[Node],
    source_layer: int,
    layer_count: int,
    pending_edges: dict[int, list[str]],
    rng: random.Random,
    config: GeneratorConfig,
) -> None:
    """Record future edge requirements into ``pending_edges``."""
    topo = config.topology
    for parent in parents:
        for _ in range(_sample_out_degree(rng, topo)):
            target = source_layer + _sample_jump(rng, topo)
            if target < layer_count:
                pending_edges[target].append(parent.node_id)


def _sample_out_degree(rng: random.Random, topo: TopologyConfig) -> int:
    return _sample_normal_int(
        rng=rng,
        mean=topo.out_degree.mean,
        variance=topo.out_degree.variance,
        minimum=topo.out_degree.minimum,
        maximum=topo.out_degree.maximum,
    )


def _sample_jump(rng: random.Random, topo: TopologyConfig) -> int:
    weights = _linear_jump_weights(topo.max_layer_jump)
    return _weighted_index(weights, rng) + 1


def _linear_jump_weights(max_layer_jump: int) -> tuple[float, ...]:
    """Normalize linearly decreasing weights [m, m-1, ..., 1] for jumps 1..m."""
    total = max_layer_jump * (max_layer_jump + 1) / 2
    return tuple(weight / total for weight in range(max_layer_jump, 0, -1))


def _weighted_index(weights: tuple[float, ...], rng: random.Random) -> int:
    r = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def _add_edge(
    graph: DirectedGraph,
    source: str,
    target: str,
) -> None:
    """Add a dependency edge unless it already exists."""
    if target in graph.successors(source):
        return
    graph.add_edge(
        Edge(
            edge_id=f"E{len(graph.edges())}",
            source=source,
            target=target,
        )
    )


# ---------------------------------------------------------------------------
# Node / cost helpers (kept from the previous version)
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str,
    layer: int,
    rng: random.Random,
    config: GeneratorConfig,
) -> Node:
    """Create one randomized node."""
    if len(config.node_types) == 1:
        node_type = config.node_types[0]
    else:
        node_type = "GPU" if rng.random() < config.gpu_ratio else "CPU"
    tokens = {}
    time_ms = None
    if node_type == "GPU":
        time_ms = _sample_normal_int(
            rng=rng,
            mean=config.resources.gpu_time_ms.mean,
            variance=config.resources.gpu_time_ms.variance,
            minimum=config.resources.gpu_time_ms.minimum,
            maximum=config.resources.gpu_time_ms.maximum,
        )
        tokens = {
            "tokens": _sample_normal_int(
                rng=rng,
                mean=config.resources.gpu_token_count.mean,
                variance=config.resources.gpu_token_count.variance,
                minimum=config.resources.gpu_token_count.minimum,
                maximum=config.resources.gpu_token_count.maximum,
            )
        }
    if node_type == "CPU":
        time_ms = _sample_normal_int(
            rng=rng,
            mean=config.resources.cpu_time_ms.mean,
            variance=config.resources.cpu_time_ms.variance,
            minimum=config.resources.cpu_time_ms.minimum,
            maximum=config.resources.cpu_time_ms.maximum,
        )

    return Node(
        node_id=node_id,
        node_type=node_type,
        layer=layer,
        tokens=tokens,
        time_ms=time_ms,
    )


def _sample_normal_int(
    rng: random.Random,
    mean: float,
    variance: float,
    minimum: int,
    maximum: int | None,
) -> int:
    stddev = variance**0.5
    sampled = rng.gauss(mean, stddev)
    value = max(minimum, int(round(sampled)))
    if maximum is not None:
        value = min(maximum, value)
    return value


def _estimate_cost(graph: DirectedGraph) -> dict[str, int]:
    """Sum tokens and CPU/GPU execution times for a graph."""
    total_tokens = sum(node.tokens.get("tokens", 0) for node in graph.nodes())
    total_cpu_time_ms = sum(
        node.time_ms or 0 for node in graph.nodes() if node.node_type == "CPU"
    )
    total_gpu_time_ms = sum(
        node.time_ms or 0 for node in graph.nodes() if node.node_type == "GPU"
    )
    return {
        "total_tokens": total_tokens,
        "total_cpu_time_ms": total_cpu_time_ms,
        "total_gpu_time_ms": total_gpu_time_ms,
    }
