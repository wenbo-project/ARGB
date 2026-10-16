from collections import Counter
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from graph_generator import (
    generate_graph,
    graph_to_dict,
    load_config,
    get_config_path,
    save_dataset,
)
from graph_generator.generator import _make_node


@pytest.mark.parametrize("name", ["default.toml", "large_graph.toml", "tree.toml"])
def test_presets_inherit_default_gpu_ratio(name):
    assert load_config(get_config_path(name)).gpu_ratio == 0.5


def test_gpu_ratio_toml_precedence_and_export(tmp_path):
    path = tmp_path / "ratio.toml"
    path.write_text("[generator]\ngpu_ratio = 0.3\n", encoding="utf-8")
    assert load_config(path).gpu_ratio == 0.3
    config = load_config(path, gpu_ratio=0.7, layer_count=3, out_degree=1)
    assert config.gpu_ratio == 0.7
    graph = generate_graph(seed=7, config=config)
    output = tmp_path / "dataset.json"
    save_dataset([graph], output, config=config)
    assert json.loads(output.read_text(encoding="utf-8"))["config"]["gpu_ratio"] == 0.7
    assert graph_to_dict(graph) == graph_to_dict(
        generate_graph(seed=7, config=config)
    )


@pytest.mark.parametrize(
    "ratio", [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, "0.3"]
)
def test_invalid_gpu_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="gpu_ratio"):
        load_config(gpu_ratio=ratio)
    with pytest.raises(ValueError, match="gpu_ratio"):
        replace(load_config(), gpu_ratio=ratio)


@pytest.mark.parametrize("node_types", [("GPU", "CPU"), ("CPU", "GPU")])
@pytest.mark.parametrize("ratio,expected_gpu_count", [(0, 0), (0.3, 30), (0.7, 70), (1, 100)])
def test_gpu_ratio_sampling_is_independent_of_type_order(node_types, ratio, expected_gpu_count):
    config = load_config(node_types=node_types, gpu_ratio=ratio)
    rng = Mock()
    rng.random.side_effect = [(i + 0.5) / 100 for i in range(100)]
    rng.gauss.return_value = 1
    counts = Counter(_make_node(f"N_0_{i}", 0, rng, config).node_type for i in range(100))
    assert counts["GPU"] == expected_gpu_count
    assert counts["CPU"] == 100 - expected_gpu_count


@pytest.mark.parametrize(
    "ratio,sample,expected", [(0, 0, "CPU"), (1, 0.999, "GPU"), (0.3, 0.299, "GPU"), (0.3, 0.3, "CPU")]
)
def test_gpu_ratio_sampling_boundaries(ratio, sample, expected):
    config = load_config(gpu_ratio=ratio)
    rng = Mock()
    rng.random.return_value = sample
    rng.gauss.return_value = 1
    assert _make_node("N_0_0", 0, rng, config).node_type == expected


@pytest.mark.parametrize("node_type", ["CPU", "GPU"])
@pytest.mark.parametrize("ratio", [0, 0.3, 1])
def test_single_node_type_takes_precedence(node_type, ratio):
    graph = generate_graph(
        seed=7, node_types=[node_type], gpu_ratio=ratio,
        first_layer_width=3, layer_count=3, out_degree=1,
    )
    assert {node.node_type for node in graph.nodes()} == {node_type}
