import pytest

from graph_generator import (
    NormalIntDistribution,
    config_to_dict,
    generate_graph,
    load_config,
    get_config_path,
)


def test_default_config_loads_and_validates():
    config = load_config()
    config.validate()
    assert isinstance(config.topology.layer_count, NormalIntDistribution)


def test_packaged_defaults_present():
    """Check the current packaged default baseline."""
    config = load_config()
    topo = config.topology
    assert topo.layer_count.mean == 40.0
    assert topo.layer_count.variance == 0.0
    assert topo.layer_count.minimum == 2
    assert topo.out_degree.mean == 0.7
    assert topo.out_degree.variance == 16.0
    assert topo.out_degree.minimum == 0
    assert topo.max_layer_jump == 1
    assert topo.first_layer_width == NormalIntDistribution(
        mean=1, variance=0, minimum=1, maximum=50
    )
    assert topo.force_converge is True
    assert config.resources.cpu_time_ms.mean == 1.0
    assert config.resources.gpu_time_ms == NormalIntDistribution(mean=1, variance=0)
    assert config.resources.gpu_token_count.mean == 1.0


def test_overrides_have_highest_precedence():
    base = load_config()
    overridden = load_config(out_degree={"mean": 5.0})
    assert base.topology.out_degree.mean == 0.7
    assert overridden.topology.out_degree.mean == 5.0
    assert overridden.topology.out_degree.variance == base.topology.out_degree.variance
    assert overridden.topology.out_degree.maximum == base.topology.out_degree.maximum


def test_scalar_layer_count_normalized_to_distribution():
    config = load_config(layer_count=12)
    dist = config.topology.layer_count
    assert isinstance(dist, NormalIntDistribution)
    assert dist.mean == 12.0
    assert dist.variance == 0.0
    assert dist.minimum == 2  # context-aware minimum for layer_count


def test_dict_layer_count_merges_fields():
    config = load_config(layer_count={"mean": 14})
    assert config.topology.layer_count.mean == 14.0
    assert config.topology.layer_count.minimum == 2  # 沿用 default.toml


def test_first_layer_width_overrides_and_export(tmp_path):
    user_toml = tmp_path / "width.toml"
    user_toml.write_text(
        "[generator.topology]\nfirst_layer_width = { mean = 3, variance = 1 }\n",
        encoding="utf-8",
    )
    config = load_config(user_toml, first_layer_width={"variance": 0})
    assert config_to_dict(config)["topology"]["first_layer_width"] == {
        "mean": 3, "variance": 0, "minimum": 1, "maximum": 50
    }


@pytest.mark.parametrize(
    "width", [{"mean": 0}, {"variance": -1}, {"minimum": 0}, {"maximum": 0}]
)
def test_invalid_first_layer_width_raises(width):
    with pytest.raises(ValueError, match="first_layer_width"):
        load_config(first_layer_width=width)


def test_budget_must_fit_minimum_first_layer_and_one_more_node():
    with pytest.raises(ValueError, match="first_layer_width.minimum"):
        load_config(first_layer_width={"minimum": 3}, max_total_nodes=3)


def test_user_toml_overrides_default(tmp_path):
    user_toml = tmp_path / "user.toml"
    user_toml.write_text(
        "[generator]\n"
        "[generator.topology]\n"
        "out_degree = { mean = 4.0 }\n",
        encoding="utf-8",
    )
    config = load_config(user_toml)
    assert config.topology.out_degree.mean == 4.0
    assert config.topology.out_degree.variance == 16.0
    assert config.topology.out_degree.minimum == 0
    assert config.topology.out_degree.maximum == 10
    overridden = load_config(user_toml, out_degree={"variance": 0})
    assert overridden.topology.out_degree.mean == 4.0
    assert overridden.topology.out_degree.variance == 0
    # 未覆盖的字段仍来自 default.toml
    assert config.topology.layer_count.mean == 40.0


def test_user_toml_without_generator_section_raises(tmp_path):
    user_toml = tmp_path / "bad.toml"
    user_toml.write_text("[other]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing \\[generator\\] section"):
        load_config(user_toml)


@pytest.mark.parametrize(
    "out_degree",
    [
        {"mean": -1},
        {"variance": -1},
        {"minimum": -1},
        {"minimum": 3, "maximum": 2},
    ],
)
def test_invalid_out_degree_raises(out_degree):
    with pytest.raises(ValueError, match="topology.out_degree"):
        load_config(out_degree=out_degree)


@pytest.mark.parametrize("out_degree", [0, 0.5, {"mean": 0, "maximum": 0}, {"mean": 0.8}])
def test_nonnegative_out_degree_loads(out_degree):
    config = load_config(out_degree=out_degree)
    assert config.topology.out_degree.minimum == 0


@pytest.mark.parametrize("field", ["layer_count", "first_layer_width", "cpu_time_ms", "gpu_time_ms", "gpu_token_count"])
def test_other_distributions_still_require_positive_values(field):
    with pytest.raises(ValueError, match=field):
        load_config(**{field: {"minimum": 0}})
    with pytest.raises(ValueError, match=field):
        load_config(**{field: 0})


@pytest.mark.parametrize(
    "out_degree",
    [3, NormalIntDistribution(mean=3, variance=0, minimum=1)],
)
def test_fixed_out_degree_generates_expected_tree(out_degree):
    config = load_config(
        layer_count=3, first_layer_width=1, out_degree=out_degree, max_layer_jump=1, force_converge=False
    )
    graph = generate_graph(seed=7, config=config)
    assert [len(graph.nodes_in_layer(layer)) for layer in range(3)] == [1, 3, 9]
    assert config.topology.out_degree.variance == 0


@pytest.mark.parametrize("max_layer_jump", [0, -1])
def test_invalid_max_layer_jump_raises(max_layer_jump):
    with pytest.raises(ValueError, match="max_layer_jump"):
        load_config(max_layer_jump=max_layer_jump)


@pytest.mark.parametrize("max_layer_jump", [1, 3])
def test_toml_only_needs_max_layer_jump(tmp_path, max_layer_jump):
    user_toml = tmp_path / "jump.toml"
    user_toml.write_text(
        f"[generator.topology]\nmax_layer_jump = {max_layer_jump}\n",
        encoding="utf-8",
    )
    config = load_config(user_toml)
    assert config.topology.max_layer_jump == max_layer_jump
    assert "jump_weights" not in config_to_dict(config)["topology"]
    graph = generate_graph(seed=7, config=config)
    by_id = {node.node_id: node for node in graph.nodes()}
    assert graph.edges()
    assert all(
        1 <= by_id[edge.target].layer - by_id[edge.source].layer <= max_layer_jump
        for edge in graph.edges()
    )


@pytest.mark.parametrize("name", ["default.toml", "large_graph.toml", "tree.toml"])
def test_packaged_presets_generate_without_weight_configuration(name):
    config = load_config(get_config_path(name))
    assert config.resources.gpu_time_ms == NormalIntDistribution(mean=1, variance=0)
    expected_width = 1
    expected_maximum = 50 if name == "default.toml" else 10
    assert config.topology.first_layer_width == NormalIntDistribution(
        mean=expected_width, variance=0, minimum=1, maximum=expected_maximum
    )
    graph = generate_graph(seed=7, config=config)
    assert len(graph.nodes_in_layer(0)) == expected_width
    assert graph.nodes()
    assert all(node.time_ms == 1 for node in graph.nodes() if node.node_type == "GPU")
    assert "jump_weights" not in config_to_dict(config)["topology"]
    if name == "tree.toml":
        assert len(graph.nodes()) == 121


def test_gpu_time_configuration_precedence_and_export(tmp_path):
    path = tmp_path / "gpu.toml"
    path.write_text(
        "[generator.resources]\ngpu_time_ms = { mean = 5, variance = 2, maximum = 10 }\n",
        encoding="utf-8",
    )
    config = load_config(path, gpu_time_ms={"mean": 7})
    assert config_to_dict(config)["resources"]["gpu_time_ms"] == {
        "mean": 7, "variance": 2, "minimum": 1, "maximum": 10
    }


@pytest.mark.parametrize(
    "gpu_time_ms", [{"mean": 0}, {"variance": -1}, {"minimum": 0}, {"maximum": 0}]
)
def test_invalid_gpu_time_raises(gpu_time_ms):
    with pytest.raises(ValueError, match="gpu_time_ms"):
        load_config(gpu_time_ms=gpu_time_ms)
