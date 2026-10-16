from collections import defaultdict, deque
from dataclasses import asdict, replace
from functools import lru_cache
import json

from graph_generator import NormalIntDistribution, load_config


@lru_cache(maxsize=1)
def default_config():
    return load_config()


@lru_cache(maxsize=64)
def small_config(seed):
    base = default_config()
    return replace(
        base,
        gpu_ratio=(seed % 5) / 4,
        topology=replace(
            base.topology,
            layer_count=NormalIntDistribution(3 + seed % 3, 0, 2),
            first_layer_width=NormalIntDistribution(1 + seed % 4, 0),
            out_degree=NormalIntDistribution(seed % 3, 1, 0, 4),
            max_layer_jump=1 + seed % 3,
            max_layer_width=6,
            max_total_nodes=24,
            force_converge=bool(seed % 2),
        ),
        resources=replace(
            base.resources,
            cpu_time_ms=NormalIntDistribution(3, 4, 1, 8),
            gpu_time_ms=NormalIntDistribution(5, 9, 2, 12),
            gpu_token_count=NormalIntDistribution(20, 25, 1, 50),
        ),
    )


def resource_totals(nodes):
    return {
        "total_tokens": sum(node.tokens.get("tokens", 0) for node in nodes),
        "total_cpu_time_ms": sum(node.time_ms or 0 for node in nodes if node.node_type == "CPU"),
        "total_gpu_time_ms": sum(node.time_ms or 0 for node in nodes if node.node_type == "GPU"),
    }


def expected_dataset(graphs, config=None, with_layer=True):
    graph_items = []
    for graph in graphs:
        nodes = []
        for node in graph.nodes():
            entry = {"node_id": node.node_id, "node_type": node.node_type}
            if with_layer:
                entry["layer"] = node.layer
            if node.tokens:
                entry["tokens"] = dict(node.tokens)
            if node.time_ms is not None:
                entry["time_ms"] = node.time_ms
            nodes.append(entry)
        graph_items.append({
            "id": graph.graph["id"],
            "estimated_cost": resource_totals(graph.nodes()),
            "nodes": nodes,
            "edges": [asdict(edge) for edge in graph.edges()],
        })
    data = {"graphs": graph_items}
    if config is not None:
        data["config"] = json.loads(json.dumps(asdict(config), allow_nan=False))
    return data


def assert_graph_indexes(graph):
    nodes, edges = graph.nodes(), graph.edges()
    by_id = {node.node_id: node for node in nodes}
    assert len(by_id) == len(nodes)
    assert len({edge.edge_id for edge in edges}) == len(edges)
    pairs = {(edge.source, edge.target) for edge in edges}
    assert len(pairs) == len(edges)
    successors = {node.node_id: set() for node in nodes}
    predecessors = {node.node_id: set() for node in nodes}
    layers = defaultdict(list)
    for node in nodes:
        assert isinstance(node.layer, int) and node.layer >= 0
        layers[node.layer].append(node)
    for edge in edges:
        assert edge.source in by_id and edge.target in by_id
        successors[edge.source].add(edge.target)
        predecessors[edge.target].add(edge.source)
    for node in nodes:
        assert set(graph.successors(node.node_id)) == successors[node.node_id]
        assert set(graph.predecessors(node.node_id)) == predecessors[node.node_id]
        assert graph.out_degree(node.node_id) == len(successors[node.node_id])
        assert graph.in_degree(node.node_id) == len(predecessors[node.node_id])
    for layer, expected in layers.items():
        assert graph.nodes_in_layer(layer) == expected
    assert graph.nodes_in_layer(max(layers, default=-1) + 1) == []
    return by_id, successors, predecessors, layers


def assert_generated_graph(graph, config):
    by_id, successors, predecessors, layers = assert_graph_indexes(graph)
    nodes = graph.nodes()
    topo = config.topology
    last_layer = max(layers)
    assert sorted(layers) == list(range(last_layer + 1))
    assert len(layers) >= 2
    if topo.layer_count.maximum is not None:
        assert len(layers) <= topo.layer_count.maximum
    assert len(nodes) <= topo.max_total_nodes
    assert len(nodes) >= len(layers[0]) + 1
    for layer, layer_nodes in layers.items():
        if layer:
            assert len(layer_nodes) <= topo.max_layer_width
        for node in layer_nodes:
            assert node.node_type in config.node_types
            assert isinstance(node.time_ms, int)
            distribution = config.resources.cpu_time_ms if node.node_type == "CPU" else config.resources.gpu_time_ms
            assert node.time_ms >= distribution.minimum
            if distribution.maximum is not None:
                assert node.time_ms <= distribution.maximum
            if node.node_type == "CPU":
                assert node.tokens == {}
            else:
                assert set(node.tokens) == {"tokens"}
                tokens = node.tokens["tokens"]
                assert isinstance(tokens, int)
                assert tokens >= config.resources.gpu_token_count.minimum
                if config.resources.gpu_token_count.maximum is not None:
                    assert tokens <= config.resources.gpu_token_count.maximum
            assert bool(predecessors[node.node_id]) == (layer > 0)
            assert bool(successors[node.node_id]) == (layer < last_layer)
    for edge in graph.edges():
        assert 1 <= by_id[edge.target].layer - by_id[edge.source].layer <= topo.max_layer_jump
    degrees = {key: len(value) for key, value in predecessors.items()}
    pending = deque(key for key, degree in degrees.items() if degree == 0)
    visited = []
    while pending:
        source = pending.popleft()
        visited.append(source)
        for target in successors[source]:
            degrees[target] -= 1
            if degrees[target] == 0:
                pending.append(target)
    assert len(visited) == len(nodes)
    assert graph.graph["estimated_cost"] == resource_totals(nodes)
    if len(config.node_types) == 2 and config.gpu_ratio in (0, 1):
        assert {node.node_type for node in nodes} == ({"GPU"} if config.gpu_ratio == 1 else {"CPU"})
    if topo.force_converge:
        assert len(layers[last_layer]) == 1
        sink = layers[last_layer][0].node_id
        reachable, pending = {sink}, [sink]
        while pending:
            for parent in predecessors[pending.pop()]:
                if parent not in reachable:
                    reachable.add(parent)
                    pending.append(parent)
        assert reachable == set(by_id)


class OnePass:
    def __init__(self, values):
        self.values = iter(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        assert self.iterations == 1
        return self

    def __next__(self):
        return next(self.values)
