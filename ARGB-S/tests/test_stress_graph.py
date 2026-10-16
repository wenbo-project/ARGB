import random

import pytest

from graph_generator import Edge, DirectedGraph, Node
from .stress_support import assert_graph_indexes


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("scenario", range(10))
def test_graph_indexes_and_insertions(seed, scenario):
    rng = random.Random(seed)
    graph = DirectedGraph(f"图_{seed}")
    for layer in range(4):
        for index in range(2 + seed % 3):
            graph.add_node(Node(f'n_{layer}_{index}_<&"', rng.choice(["CPU", "GPU"]), layer))
    source, peer = graph.nodes_in_layer(0)[:2]
    target = graph.nodes_in_layer(1)[0]
    reserved = {
        (source.node_id, source.node_id),
        (target.node_id, source.node_id),
        (source.node_id, peer.node_id),
    }
    for parent in graph.nodes():
        for child in graph.nodes():
            if (parent.node_id, child.node_id) not in reserved and rng.random() < 0.12:
                graph.add_edge(Edge(f"e_{len(graph.edges())}", parent.node_id, child.node_id))
    if target.node_id not in graph.successors(source.node_id):
        graph.add_edge(Edge("initial", source.node_id, target.node_id))
    before_nodes, before_edges = graph.nodes(), graph.edges()
    if scenario == 0:
        graph.nodes().clear()
        graph.edges().clear()
        graph.nodes_in_layer(0).clear()
        graph.successors(source.node_id).clear()
        graph.predecessors(target.node_id).clear()
    elif scenario in (4, 5, 6):
        parent, child = [
            (source.node_id, source.node_id),
            (target.node_id, source.node_id),
            (source.node_id, peer.node_id),
        ][scenario - 4]
        edge = Edge("accepted", parent, child)
        graph.add_edge(edge)
        assert graph.nodes() == before_nodes
        assert graph.edges() == before_edges + [edge]
        assert child in graph.successors(parent)
        assert parent in graph.predecessors(child)
        with pytest.raises(ValueError, match="duplicate dependency edge"):
            graph.add_edge(Edge("duplicate_accepted", parent, child))
        assert graph.edges() == before_edges + [edge]
        assert_graph_indexes(graph)
        return
    else:
        operation = {
            1: lambda: graph.add_node(source),
            2: lambda: graph.add_node(Node("invalid", "CPU", -1 - seed)),
            3: lambda: graph.add_edge(Edge(before_edges[0].edge_id, source.node_id, target.node_id)),
            7: lambda: graph.add_edge(Edge("invalid", f"missing_{seed}", target.node_id)),
            8: lambda: graph.add_edge(Edge("invalid", source.node_id, f"missing_{seed}")),
            9: lambda: graph.add_edge(Edge("duplicate_pair", source.node_id, target.node_id)),
        }[scenario]
        with pytest.raises(ValueError):
            operation()
    assert graph.nodes() == before_nodes
    assert graph.edges() == before_edges
    assert_graph_indexes(graph)


@pytest.mark.parametrize("layer", [0.5, 1.0, True, False, "1", None, float("nan"), float("inf")])
def test_noninteger_layer_rejected_without_mutating_graph(layer):
    graph = DirectedGraph("invalid")
    graph.add_node(Node("valid", "CPU", 0))
    with pytest.raises(ValueError):
        graph.add_node(Node("invalid", "CPU", layer))
    assert graph.nodes() == [Node("valid", "CPU", 0)]
    assert_graph_indexes(graph)
