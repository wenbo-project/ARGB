import json

import pytest

from graph_generator import (
    DirectedGraph, Edge, Node, generate_graph, graphs_to_dataset, load_config,
    save_dataset, save_dataset_no_layer,
)
from graph_generator.export import graph_to_dict


def test_default_and_custom_graph_identifiers():
    config = load_config(layer_count=3, out_degree=1)
    default = generate_graph(config=config, seed=7)
    custom = generate_graph(id="custom", config=config, seed=7)
    assert isinstance(default, DirectedGraph)
    assert default.graph["id"] == "000001"
    assert custom.graph["id"] == "custom"
    assert default.nodes() == custom.nodes()
    assert default.edges() == custom.edges()


def test_public_types_and_export_schema():
    graph = DirectedGraph(id="custom")
    graph.add_node(Node(node_id="cpu", node_type="CPU", layer=0, time_ms=3))
    graph.add_node(Node(node_id="gpu", node_type="GPU", layer=1, tokens={"tokens": 5}, time_ms=4))
    graph.add_edge(Edge(edge_id="dependency", source="cpu", target="gpu"))
    expected = {
        "id": "custom",
        "estimated_cost": {},
        "nodes": [
            {"node_id": "cpu", "node_type": "CPU", "layer": 0, "time_ms": 3},
            {"node_id": "gpu", "node_type": "GPU", "layer": 1, "tokens": {"tokens": 5}, "time_ms": 4},
        ],
        "edges": [{"edge_id": "dependency", "source": "cpu", "target": "gpu"}],
    }
    assert graph_to_dict(graph) == expected
    assert graphs_to_dataset([graph]) == {"graphs": [expected]}


@pytest.mark.parametrize("pairs", [
    [("a", "a")],
    [("a", "b"), ("b", "a")],
    [("c", "a")],
    [("a", "b"), ("b", "c"), ("c", "a")],
], ids=["self", "same_layer", "reverse", "closed_path"])
@pytest.mark.parametrize("with_layer", [False, True])
def test_export_preserves_general_directed_edges(tmp_path, pairs, with_layer):
    graph = DirectedGraph(id="custom")
    for node_id, layer in [("a", 0), ("b", 0), ("c", 1)]:
        graph.add_node(Node(node_id=node_id, node_type="CPU", layer=layer))
    expected_edges = []
    for index, (source, target) in enumerate(pairs):
        edge_id = f"e{index}"
        graph.add_edge(Edge(edge_id=edge_id, source=source, target=target))
        expected_edges.append({"edge_id": edge_id, "source": source, "target": target})
    path = tmp_path / "graphs.json"
    writer = save_dataset if with_layer else save_dataset_no_layer
    writer([graph], path)
    exported = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    assert exported["id"] == "custom"
    assert exported["edges"] == expected_edges
    assert [node["node_id"] for node in exported["nodes"]] == ["a", "b", "c"]
    assert all(("layer" in node) == with_layer for node in exported["nodes"])
    assert [node.layer for node in graph.nodes()] == [0, 0, 1]
