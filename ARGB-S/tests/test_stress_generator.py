from dataclasses import replace

import pytest

from graph_generator import NormalIntDistribution, generate_graph, graph_to_dict
from .stress_support import assert_generated_graph, small_config


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("layer_count", [2, 3, 5, 8, 12])
@pytest.mark.parametrize("degree_profile", [0, 1, 2, 3])
@pytest.mark.parametrize("budget_profile", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("force_converge", [False, True])
def test_generated_graph_invariants(seed, layer_count, degree_profile, budget_profile, force_converge):
    base = small_config(seed)
    entry_width = 1 + seed % 4
    budget = [entry_width + 1, entry_width + 2, entry_width + 5, 24, 64][budget_profile]
    degree = [
        NormalIntDistribution(0, 0, 0, 0),
        NormalIntDistribution(1, 0, 0, 1),
        NormalIntDistribution(0.7, 16, 0, 10),
        NormalIntDistribution(3, 4, 0, 8),
    ][degree_profile]
    config = replace(
        base,
        node_types=[("CPU",), ("GPU",), ("CPU", "GPU"), ("GPU", "CPU")][seed % 4],
        gpu_ratio=[0, 0.2, 0.5, 0.8, 1][(seed + budget_profile) % 5],
        topology=replace(
            base.topology,
            layer_count=NormalIntDistribution(layer_count, 0, 2, layer_count),
            first_layer_width=NormalIntDistribution(entry_width, 0, 1, entry_width),
            out_degree=degree,
            max_layer_jump=[1, 2, 3, 8][(seed + layer_count) % 4],
            max_layer_width=[1, 2, 5, 12][(seed + degree_profile) % 4],
            max_total_nodes=budget,
            force_converge=force_converge,
        ),
    )
    graph = generate_graph(id=f"case_{seed}_图", seed=seed, config=config)
    assert_generated_graph(graph, config)
    assert len(graph.nodes_in_layer(0)) == entry_width
    assert max(node.layer for node in graph.nodes()) < layer_count
    assert graph_to_dict(graph) == graph_to_dict(
        generate_graph(id=f"case_{seed}_图", seed=seed, config=config)
    )
