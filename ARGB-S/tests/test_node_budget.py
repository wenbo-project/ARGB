import pytest

from graph_generator import (
    NormalIntDistribution,
    generate_graph,
    graph_to_dict,
    load_config,
)


def assert_all_nodes_reach_sink(graph):
    sinks = [node.node_id for node in graph.nodes() if graph.out_degree(node.node_id) == 0]
    assert len(sinks) == 1
    visited = set(sinks)
    pending = list(sinks)
    while pending:
        for parent in graph.predecessors(pending.pop()):
            if parent not in visited:
                visited.add(parent)
                pending.append(parent)
    assert visited == {node.node_id for node in graph.nodes()}


@pytest.mark.parametrize("budget,width,layer_count,expected_widths", [
    (2, 1, 10, [1, 1]),
    (6, 5, 10, [5, 1]),
    (7, 5, 10, [5, 1, 1]),
    (10, 1, 10, [1, 3, 5, 1]),
    (4, 1, 3, [1, 2, 1]),
    (2, 1, 2, [1, 1]),
])
def test_converge_uses_reserved_slot(budget, width, layer_count, expected_widths):
    config = load_config(
        first_layer_width=width, layer_count=layer_count, out_degree=3,
        max_layer_jump=1, max_total_nodes=budget, force_converge=True,
    )
    graph = generate_graph(seed=7, config=config)
    last_layer = max(node.layer for node in graph.nodes())
    assert [len(graph.nodes_in_layer(layer)) for layer in range(last_layer + 1)] == expected_widths
    assert len(graph.nodes()) == budget
    assert_all_nodes_reach_sink(graph)
    assert all(
        graph.in_degree(node.node_id) >= 1
        for node in graph.nodes() if node.layer > 0
    )
    assert graph_to_dict(graph) == graph_to_dict(
        generate_graph(seed=7, config=config)
    )


@pytest.mark.parametrize("force_converge,expected_widths", [
    (False, [1, 3, 6]),
    (True, [1, 3, 5, 1]),
])
def test_full_budget_without_converge_and_reserved_budget_with_converge(
    force_converge, expected_widths,
):
    graph = generate_graph(
        seed=7, first_layer_width=1, layer_count=4, out_degree=3,
        max_layer_jump=1, max_total_nodes=10, force_converge=force_converge,
    )
    last_layer = max(node.layer for node in graph.nodes())
    assert [len(graph.nodes_in_layer(layer)) for layer in range(last_layer + 1)] == expected_widths
    assert len(graph.nodes()) == 10


@pytest.mark.parametrize("width", [
    5,
    {"mean": 5, "variance": 0, "minimum": 1, "maximum": 10},
    NormalIntDistribution(mean=5, variance=0),
])
def test_fixed_entry_without_room_for_converge_is_rejected_during_loading(width):
    with pytest.raises(ValueError, match="reserve one converge node.*first_layer_width"):
        load_config(
            first_layer_width=width, max_total_nodes=5, force_converge=True,
        )


def test_invalid_fixed_entry_in_toml_is_rejected_during_loading(tmp_path):
    path = tmp_path / "budget.toml"
    path.write_text(
        "[generator.topology]\n"
        "first_layer_width = { mean = 5, variance = 0 }\n"
        "max_total_nodes = 5\nforce_converge = true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserve one converge node"):
        load_config(path)


@pytest.mark.parametrize("distribution,budget,expected_width", [
    (NormalIntDistribution(mean=5, variance=0, maximum=3), 4, 3),
    (NormalIntDistribution(mean=1, variance=0, minimum=3), 4, 3),
    (NormalIntDistribution(mean=2.6, variance=0), 4, 3),
])
def test_fixed_entry_validation_uses_sampler_rounding_and_bounds(
    distribution, budget, expected_width,
):
    config = load_config(
        first_layer_width=distribution, max_total_nodes=budget,
        layer_count=10, force_converge=True,
    )
    graph = generate_graph(seed=7, config=config)
    assert len(graph.nodes_in_layer(0)) == expected_width
    assert len(graph.nodes()) == budget
    assert_all_nodes_reach_sink(graph)


@pytest.mark.parametrize("budget", [2, 7, 20])
@pytest.mark.parametrize("force_converge", [False, True])
@pytest.mark.parametrize("max_layer_jump", [1, 3])
def test_strict_limit_under_random_out_degrees(budget, force_converge, max_layer_jump):
    config = load_config(
        first_layer_width=1, layer_count=12, max_total_nodes=budget,
        force_converge=force_converge, max_layer_jump=max_layer_jump,
        out_degree={"mean": 2, "variance": 4},
    )
    for seed in (0, 7, 13, 42, 99):
        graph = generate_graph(seed=seed, config=config)
        assert len(graph.nodes()) <= budget
        if force_converge:
            assert_all_nodes_reach_sink(graph)
