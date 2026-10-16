from dataclasses import replace

import pytest

from graph_generator import (
    generate_graph,
    load_config,
    get_config_path,
)


@pytest.mark.parametrize("overrides", [
    {"layr_count": 3},
    {"layr_count": None},
    {"topology": {"layr_count": 3}},
    {"resources": {"gpu_tokens": 3}},
    {"out_degree": {"varaince": 0}},
    {"resources": {"cpu_time_ms": {"meen": 3}}},
])
def test_unknown_override_fields_raise(overrides):
    with pytest.raises(ValueError, match="unknown fields.*allowed fields"):
        load_config(**overrides)


@pytest.mark.parametrize("contents", [
    "[generator]\nnode_typess = ['CPU']\n",
    "[generator.topology]\nlayr_count = 3\n",
    "[generator.resources]\ngpu_tokens = 3\n",
    "[generator.topology]\nout_degree = { mean = 3, varaince = 0 }\n",
    "[generator.resources]\ncpu_time_ms = { meen = 3 }\n",
    "[generator]\n[run]\ncount = 3\n",
])
def test_unknown_toml_fields_raise(tmp_path, contents):
    path = tmp_path / "bad.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields.*allowed fields"):
        load_config(path)


def test_invalid_toml_cannot_be_hidden_by_valid_override(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        "[generator.topology]\nout_degree = { mean = 3, varaince = 0 }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="varaince"):
        load_config(path, out_degree=2)


@pytest.mark.parametrize("node_types", [
    [], ["TPU"], ["gpu"], ["Gpu"], [" GPU"], ["GPU "],
    ["GPU", "GPU"], ["CPU", "GPU", "CPU"], [1], "GPU",
])
def test_invalid_node_types_raise(node_types):
    with pytest.raises(ValueError, match="node_types"):
        load_config(node_types=node_types)


@pytest.mark.parametrize("node_types", [
    ("CPU",), ("GPU",), ("CPU", "GPU"), ("GPU", "CPU"),
])
def test_valid_node_types_load(node_types):
    assert load_config(node_types=node_types).node_types == node_types


@pytest.mark.parametrize("node_types", [
    (), ("TPU",), ("CPU", "CPU"), ("GPU ",), ("gpu",),
])
def test_direct_config_construction_validates_node_types(node_types):
    with pytest.raises(ValueError, match="node_types"):
        replace(load_config(), node_types=node_types)


@pytest.mark.parametrize("kwargs", [
    {"config_path": get_config_path()},
    {"out_degree": 10},
    {"max_total_nodes": 500},
    {"out_degree": None},
    {"config_path": get_config_path(), "out_degree": 10},
])
def test_config_object_cannot_be_mixed_with_other_config_sources(kwargs):
    with pytest.raises(ValueError, match="config cannot be combined"):
        generate_graph(config=load_config(), **kwargs)


def test_config_object_still_accepts_id_and_seed():
    config = load_config(layer_count=3)
    graph = generate_graph(id="custom", seed=7, config=config)
    assert graph.graph["id"] == "custom"
    assert len(graph.nodes_in_layer(2)) == 1
