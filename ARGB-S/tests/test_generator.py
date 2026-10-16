from collections import Counter
from unittest.mock import Mock

import pytest

from graph_generator import NormalIntDistribution, generate_graph, load_config
from graph_generator.generator import _sample_jump, _sample_out_degree


@pytest.mark.parametrize(
    "max_layer_jump,expected_counts",
    [(1, [60]), (2, [40, 20]), (3, [30, 20, 10]), (4, [24, 18, 12, 6])],
)
def test_jump_sampling_has_linear_probabilities(max_layer_jump, expected_counts):
    config = load_config(max_layer_jump=max_layer_jump)
    # Evenly spaced quantiles check each probability interval without random noise.
    rng = Mock()
    rng.random.side_effect = [(i + 0.5) / 60 for i in range(60)]
    counts = Counter(_sample_jump(rng, config.topology) for _ in range(60))
    assert counts == dict(enumerate(expected_counts, start=1))


@pytest.mark.parametrize("sample,expected", [(-3, 0), (0.2, 0), (0.8, 1), (2.2, 2), (20, 10)])
def test_out_degree_rounding_and_clamping(sample, expected):
    config = load_config(out_degree={"mean": 0.8, "variance": 1})
    rng = Mock()
    rng.gauss.return_value = sample
    assert _sample_out_degree(rng, config.topology) == expected


def test_zero_degree_repairs_do_not_increase_layer_width(monkeypatch):
    sampled = iter([2, 0, 1, 0, 1] + [2] * 4)
    monkeypatch.setattr(
        "graph_generator.generator._sample_out_degree", lambda rng, topo: next(sampled, 0)
    )
    graph = generate_graph(
        seed=7, first_layer_width=5, layer_count=3, max_layer_jump=1, force_converge=False
    )
    assert [len(graph.nodes_in_layer(layer)) for layer in range(3)] == [5, 4, 8]
    assert [graph.out_degree(f"N_0_{i}") for i in range(5)] == [2, 1, 1, 1, 1]
    for node_id in ("N_0_1", "N_0_3"):
        child_id = graph.successors(node_id)[0]
        assert child_id.startswith("N_1_")
        assert graph.in_degree(child_id) >= 2


@pytest.mark.parametrize("force_converge", [False, True])
def test_all_zero_degrees_preserve_connected_layers(force_converge):
    graph = generate_graph(
        seed=7, first_layer_width=5, layer_count=4, out_degree=0,
        max_layer_jump=1, force_converge=force_converge,
    )
    assert [len(graph.nodes_in_layer(layer)) for layer in range(4)] == [5, 1, 1, 1]
    assert len(graph.edges()) == 7
    for node in graph.nodes():
        assert graph.out_degree(node.node_id) == (0 if node.layer == 3 else 1)
        if node.layer > 0:
            assert graph.in_degree(node.node_id) >= 1


@pytest.mark.parametrize("max_total_nodes", [7, 100])
@pytest.mark.parametrize("force_converge", [False, True])
def test_unrealized_jumps_get_next_layer_successors(monkeypatch, max_total_nodes, force_converge):
    monkeypatch.setattr("graph_generator.generator._sample_jump", lambda rng, topo: 3)
    graph = generate_graph(
        seed=7, first_layer_width=5, layer_count=4, out_degree=1,
        max_layer_jump=3, max_total_nodes=max_total_nodes, force_converge=force_converge,
    )
    by_id = {node.node_id: node for node in graph.nodes()}
    last_layer = max(node.layer for node in graph.nodes())
    assert len(by_id) <= max_total_nodes
    for node in by_id.values():
        if node.layer < last_layer:
            assert graph.out_degree(node.node_id) >= 1
        else:
            assert graph.out_degree(node.node_id) == 0
        if node.layer > 0:
            assert graph.in_degree(node.node_id) >= 1
    pairs = [(edge.source, edge.target) for edge in graph.edges()]
    assert len(pairs) == len(set(pairs))
    assert all(1 <= by_id[target].layer - by_id[source].layer <= 3 for source, target in pairs)
    if force_converge:
        assert len(graph.nodes_in_layer(last_layer)) == 1


def test_subunit_mean_can_shrink_reproducibly():
    from graph_generator import graph_to_dict

    config = load_config(
        first_layer_width=10, layer_count=8, max_layer_jump=1, force_converge=False,
        out_degree={"mean": 0.8, "variance": 0.64},
    )
    first = generate_graph(seed=7, config=config)
    second = generate_graph(seed=7, config=config)
    assert graph_to_dict(first) == graph_to_dict(second)
    widths = [len(first.nodes_in_layer(layer)) for layer in range(8)]
    assert any(right < left for left, right in zip(widths, widths[1:]))
    assert all(first.out_degree(node.node_id) >= 1 for node in first.nodes() if node.layer < 7)


def test_edges_point_forward():
    graph = generate_graph(id="t", seed=1)
    by_id = {node.node_id: node for node in graph.nodes()}
    for edge in graph.edges():
        assert by_id[edge.source].layer < by_id[edge.target].layer


def test_no_duplicate_or_self_edges():
    graph = generate_graph(id="t", seed=1)
    seen = set()
    for edge in graph.edges():
        assert edge.source != edge.target
        key = (edge.source, edge.target)
        assert key not in seen
        seen.add(key)


def test_non_entry_nodes_have_in_degree():
    graph = generate_graph(id="t", seed=1)
    entry_ids = {node.node_id for node in graph.nodes() if node.layer == 0}
    for node in graph.nodes():
        if node.node_id in entry_ids:
            continue
        assert graph.in_degree(node.node_id) >= 1


def test_seed_reproduces():
    a = generate_graph(id="a", seed=99)
    b = generate_graph(id="a", seed=99)
    assert len(a.nodes()) == len(b.nodes())
    assert len(a.edges()) == len(b.edges())


def test_force_converge_last_layer_single_node():
    config = load_config(
        layer_count=6,
        max_layer_jump=1,
        out_degree={"mean": 2.0, "variance": 0.0},
        max_total_nodes=1000,
    )
    graph = generate_graph(id="t", seed=3, config=config)
    last_layer = max(node.layer for node in graph.nodes())
    last_nodes = [node for node in graph.nodes() if node.layer == last_layer]
    assert len(last_nodes) == 1


def test_max_total_nodes_respected():
    config = load_config(
        out_degree={"mean": 5.0, "variance": 0.0},
        max_layer_width=100,
        max_total_nodes=50,
    )
    graph = generate_graph(id="t", seed=5, config=config)
    assert len(graph.nodes()) <= config.topology.max_total_nodes


def test_pure_tree_shape():
    """Fixed out-degree and adjacent-layer edges produce a k-ary tree."""
    config = load_config(
        first_layer_width=1,
        layer_count=5,
        out_degree={"mean": 3.0, "variance": 0.0, "minimum": 1, "maximum": 3},
        max_layer_jump=1,
        max_layer_width=10000,
        max_total_nodes=100000,
        force_converge=False,
    )
    graph = generate_graph(id="t", seed=2, config=config)
    layer1 = [node for node in graph.nodes() if node.layer == 1]
    layer2 = [node for node in graph.nodes() if node.layer == 2]
    assert len(layer1) == 3
    assert len(layer2) == 9


@pytest.mark.parametrize(
    "max_layer_width,max_total_nodes,expected_widths",
    [
        (100, 100, [1, 3, 9, 27]),
        (4, 100, [1, 3, 4, 4]),
        (100, 10, [1, 3, 6]),
    ],
)
def test_layer_width_uses_pending_edges_and_limits(
    max_layer_width, max_total_nodes, expected_widths
):
    config = load_config(
        first_layer_width=1,
        layer_count=4,
        out_degree={"mean": 3, "variance": 0},
        max_layer_jump=1,
        max_layer_width=max_layer_width,
        max_total_nodes=max_total_nodes,
        force_converge=False,
    )
    graph = generate_graph(seed=7, config=config)
    assert [
        len(graph.nodes_in_layer(layer))
        for layer in range(max(node.layer for node in graph.nodes()) + 1)
    ] == expected_widths
    for node in graph.nodes():
        if node.layer > 0:
            assert graph.in_degree(node.node_id) >= 1


def test_empty_pending_layer_keeps_one_node(monkeypatch):
    monkeypatch.setattr("graph_generator.generator._sample_jump", lambda rng, topo: 2)
    config = load_config(
        first_layer_width=1,
        layer_count=4,
        out_degree={"mean": 1, "variance": 0},
        max_layer_jump=2,
        force_converge=False,
    )
    graph = generate_graph(seed=7, config=config)
    assert [len(graph.nodes_in_layer(layer)) for layer in range(4)] == [1, 1, 1, 1]
    for node in graph.nodes():
        if node.layer > 0:
            assert graph.in_degree(node.node_id) >= 1


def test_dataset_config_recording():
    from graph_generator import graphs_to_dataset

    config = load_config(layer_count=4, out_degree={"mean": 2.0})
    graph = generate_graph(id="t", seed=1, config=config)
    dataset = graphs_to_dataset([graph], config=config)
    assert dataset["config"]["topology"]["out_degree"]["mean"] == 2.0
    assert dataset["config"]["topology"]["layer_count"]["mean"] == 4.0


@pytest.mark.parametrize(
    "width", [3, {"mean": 3}, NormalIntDistribution(mean=3, variance=0)]
)
def test_sampled_first_layer_is_counted_in_budget(width):
    config = load_config(
        first_layer_width=width,
        layer_count=4,
        out_degree=2,
        max_layer_jump=1,
        max_total_nodes=5,
        force_converge=False,
    )
    graph = generate_graph(seed=7, config=config)
    assert [len(graph.nodes_in_layer(layer)) for layer in range(3)] == [3, 2, 0]
    assert len(graph.nodes()) == 5


@pytest.mark.parametrize("sample,expected_width", [(-20, 2), (4, 4), (20, 6)])
def test_first_layer_is_sampled_once_and_clamped(monkeypatch, sample, expected_width):
    import random

    original_gauss = random.Random.gauss
    width_samples = []

    def controlled_gauss(rng, mean, stddev):
        if mean == 7 and stddev == 2:
            width_samples.append(sample)
            return sample
        return original_gauss(rng, mean, stddev)

    monkeypatch.setattr(random.Random, "gauss", controlled_gauss)
    graph = generate_graph(
        seed=7,
        layer_count=3,
        first_layer_width={"mean": 7, "variance": 4, "minimum": 2, "maximum": 6},
    )
    assert len(graph.nodes_in_layer(0)) == expected_width
    assert width_samples == [sample]


def test_budget_must_fit_actual_first_layer_and_one_more_node(monkeypatch):
    import random

    monkeypatch.setattr(random.Random, "gauss", lambda rng, mean, stddev: mean)
    config = load_config(
        first_layer_width={"mean": 5, "variance": 1}, max_total_nodes=5
    )
    with pytest.raises(ValueError, match="sampled first_layer_width"):
        generate_graph(seed=7, config=config)


@pytest.mark.parametrize("node_types", [("CPU",), ("GPU",), ("CPU", "GPU")])
def test_cpu_gpu_times_export_and_totals(tmp_path, node_types):
    import json

    from graph_generator import save_dataset, visualize

    config = load_config(
        node_types=node_types,
        layer_count=3,
        first_layer_width=20,
        cpu_time_ms=3,
        gpu_time_ms={"mean": 7},
        gpu_token_count=11,
    )
    graph = generate_graph(seed=7, config=config)
    nodes = graph.nodes()
    assert {node.node_type for node in nodes} == set(node_types)
    gpu_count = sum(node.node_type == "GPU" for node in nodes)
    cpu_count = len(nodes) - gpu_count
    for node in nodes:
        assert node.time_ms == (7 if node.node_type == "GPU" else 3)
        assert node.tokens == ({"tokens": 11} if node.node_type == "GPU" else {})
    cost = graph.graph["estimated_cost"]
    assert set(cost) == {"total_tokens", "total_cpu_time_ms", "total_gpu_time_ms"}
    assert cost["total_gpu_time_ms"] == gpu_count * 7
    assert cost["total_cpu_time_ms"] == cpu_count * 3
    assert cost["total_tokens"] == gpu_count * 11

    json_path = tmp_path / "times.json"
    html_path = tmp_path / "times.html"
    save_dataset([graph], json_path, config=config)
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(exported["config"]["resources"]) == {
        "cpu_time_ms", "gpu_time_ms", "gpu_token_count"
    }
    for edge in exported["graphs"][0]["edges"]:
        assert set(edge) == {"edge_id", "source", "target"}
    for node in exported["graphs"][0]["nodes"]:
        assert node["time_ms"] == (7 if node["node_type"] == "GPU" else 3)
    visualize(input_path=json_path, output_path=html_path)
    html = html_path.read_text(encoding="utf-8")
    assert f"gpu ms: {gpu_count * 7}" in html
    assert f"cpu ms: {cpu_count * 3}" in html


def test_gpu_time_distribution_bounds_and_seed():
    from graph_generator import graph_to_dict

    config = load_config(
        node_types=("GPU",),
        first_layer_width=20,
        layer_count=3,
        gpu_time_ms={"mean": 5, "variance": 100, "minimum": 3, "maximum": 7},
    )
    first = generate_graph(seed=7, config=config)
    second = generate_graph(seed=7, config=config)
    assert graph_to_dict(first) == graph_to_dict(second)
    times = {node.time_ms for node in first.nodes()}
    assert min(times) == 3
    assert max(times) == 7
