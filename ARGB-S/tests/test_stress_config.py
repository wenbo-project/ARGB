from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from graph_generator import NormalIntDistribution, RunConfig, config_to_dict, load_config
from .stress_support import default_config


DISTRIBUTIONS = [
    ("topology", "layer_count", 2),
    ("topology", "out_degree", 0),
    ("topology", "first_layer_width", 1),
    ("resources", "cpu_time_ms", 1),
    ("resources", "gpu_time_ms", 1),
    ("resources", "gpu_token_count", 1),
]


@pytest.mark.parametrize("section,field,minimum", DISTRIBUTIONS)
@pytest.mark.parametrize("style", ["scalar", "flat", "nested", "object"])
@pytest.mark.parametrize("value_index", range(50))
def test_distribution_override_forms(section, field, minimum, style, value_index):
    base = default_config()
    value = minimum + value_index / 3
    original = getattr(getattr(base, section), field)
    if style == "scalar":
        argument = value
        distribution = NormalIntDistribution(value, 0, minimum)
    elif style == "object":
        distribution = NormalIntDistribution(value, value_index % 7, minimum, 100)
        argument = distribution
    else:
        argument = {"mean": value}
        distribution = replace(original, mean=value)
    overrides = {section: {field: argument}} if style == "nested" else {field: argument}
    snapshot = deepcopy(overrides)
    config = load_config(**overrides)
    expected = replace(base, **{section: replace(getattr(base, section), **{field: distribution})})
    assert config == expected
    assert overrides == snapshot
    assert config_to_dict(config) == json.loads(json.dumps(asdict(expected), allow_nan=False))
    assert load_config(**overrides) == config


@pytest.mark.parametrize("value_index", range(20))
@pytest.mark.parametrize("profile", range(5))
@pytest.mark.parametrize("mode", range(8))
def test_toml_and_keyword_precedence(tmp_path, value_index, profile, mode):
    base = default_config()
    path = tmp_path / "配置.toml"
    file_cpu = NormalIntDistribution(2 + value_index, profile, 1, 100)
    path.write_text(
        '[generator]\nnode_types = ["CPU", "GPU"]\n'
        f'gpu_ratio = {profile / 4}\n'
        '[generator.topology]\nmax_layer_width = 15\n'
        '[generator.resources]\n'
        f'cpu_time_ms = {{ mean = {file_cpu.mean}, variance = {profile}, minimum = 1, maximum = 100 }}\n',
        encoding="utf-8",
    )
    expected = replace(
        base, node_types=("CPU", "GPU"), gpu_ratio=profile / 4,
        topology=replace(base.topology, max_layer_width=15),
        resources=replace(base.resources, cpu_time_ms=file_cpu),
    )
    value = value_index + 30
    if mode == 0:
        overrides = {"cpu_time_ms": value}
        expected = replace(expected, resources=replace(expected.resources, cpu_time_ms=NormalIntDistribution(value, 0)))
    elif mode in (1, 2):
        overrides = {"cpu_time_ms": {"mean": value}} if mode == 1 else {"resources": {"cpu_time_ms": {"mean": value}}}
        expected = replace(expected, resources=replace(expected.resources, cpu_time_ms=replace(file_cpu, mean=value)))
    elif mode == 3:
        overrides = {"out_degree": value_index / 10}
        expected = replace(expected, topology=replace(expected.topology, out_degree=NormalIntDistribution(value_index / 10, 0, 0)))
    elif mode == 4:
        overrides = {"topology": {"max_layer_jump": value_index + 1}}
        expected = replace(expected, topology=replace(expected.topology, max_layer_jump=value_index + 1))
    elif mode == 5:
        overrides = {"gpu_ratio": value_index / 19}
        expected = replace(expected, gpu_ratio=value_index / 19)
    elif mode == 6:
        overrides = {"node_types": ["GPU"] if value_index % 2 else ["CPU"]}
        expected = replace(expected, node_types=tuple(overrides["node_types"]))
    else:
        overrides = {"force_converge": bool(value_index % 2)}
        expected = replace(expected, topology=replace(expected.topology, force_converge=bool(value_index % 2)))
    snapshot = deepcopy(overrides)
    assert load_config(config_path=path, **overrides) == expected
    assert overrides == snapshot
    assert load_config(config_path=path) == replace(
        base, node_types=("CPU", "GPU"), gpu_ratio=profile / 4,
        topology=replace(base.topology, max_layer_width=15),
        resources=replace(base.resources, cpu_time_ms=file_cpu),
    )


@pytest.mark.parametrize("section,field,minimum", DISTRIBUTIONS)
@pytest.mark.parametrize("scenario", range(10))
@pytest.mark.parametrize("context", range(10))
def test_invalid_distributions_rejected(section, field, minimum, scenario, context):
    base = default_config()
    values = {"mean": minimum + context + 2, "variance": context, "minimum": minimum, "maximum": 100}
    invalid = [
        ("mean", -1), ("mean", float("nan")), ("mean", float("inf")),
        ("variance", -1), ("variance", float("nan")), ("variance", float("inf")),
        ("minimum", -1), ("maximum", minimum - 1),
        ("minimum", minimum + 0.5), ("maximum", 100.5),
    ][scenario]
    values[invalid[0]] = invalid[1]
    with pytest.raises(ValueError):
        replace(base, **{section: replace(getattr(base, section), **{field: NormalIntDistribution(**values)})})
    with pytest.raises(ValueError):
        load_config(**{field: values, "gpu_ratio": context / 10})


INVALID_FIELDS = [
    ("max_layer_jump", [0, -1, 0.5, 1.5, True, False, "4", None, float("nan"), float("inf")]),
    ("max_layer_width", [0, -1, 0.5, 1.5, True, False, "4", None, float("nan"), float("inf")]),
    ("max_total_nodes", [0, -1, 1, 2.5, True, False, "8", None, float("nan"), float("inf")]),
    ("force_converge", [-1, 0, 1, 2, float("nan"), float("inf"), "false", "true", [], {}]),
    ("gpu_ratio", [-1, 1.01, float("nan"), float("inf"), -float("inf"), True, False, "0.5", [], {}]),
    ("node_types", [[], ["CPU", "CPU"], ["GPU", "GPU"], ["TPU"], ["cpu"], ["CPU", "TPU"], [1], "CPU", {}, None]),
    ("graph_count", [-1, -10, 0.5, 2.5, True, False, "4", None, float("nan"), float("inf")]),
    ("thread_count", [0, -1, 0.5, 2.5, True, False, "4", None, float("nan"), float("inf")]),
]


@pytest.mark.parametrize("field,bad_values", INVALID_FIELDS, ids=[item[0] for item in INVALID_FIELDS])
@pytest.mark.parametrize("bad_index", range(10))
@pytest.mark.parametrize("profile", range(5))
def test_invalid_control_fields_rejected(field, bad_values, bad_index, profile):
    base = default_config()
    base = replace(base, gpu_ratio=profile / 4, topology=replace(base.topology, max_layer_jump=profile + 1))
    value = bad_values[bad_index]
    if field in ("graph_count", "thread_count"):
        with pytest.raises(ValueError):
            RunConfig(generator=base, **{field: value})
    elif field in ("gpu_ratio", "node_types"):
        with pytest.raises(ValueError):
            replace(base, **{field: value})
        if value is not None:
            with pytest.raises(ValueError):
                load_config(**{field: value})
    else:
        with pytest.raises(ValueError):
            replace(base.topology, **{field: value})
        if value is not None:
            with pytest.raises(ValueError):
                load_config(**{field: value})


@pytest.mark.parametrize("section,field,value,other_field,other_value", [
    ("topology", "max_layer_jump", 4, "max_layer_width", 12),
    ("resources", "cpu_time_ms", 7, "gpu_time_ms", 9),
])
@pytest.mark.parametrize("nested_first", [False, True])
@pytest.mark.parametrize("collision", [False, True])
def test_nested_and_flat_overrides_are_order_independent(section, field, value, other_field, other_value, nested_first, collision):
    nested = {other_field: other_value}
    if collision:
        nested[field] = value + 1
    items = [(section, nested), (field, value)]
    overrides = dict(items if nested_first else reversed(items))
    snapshot = deepcopy(overrides)
    config = load_config(**overrides)
    assert overrides == snapshot
    owner = getattr(config, section)
    assert getattr(owner, field) == (NormalIntDistribution(value, 0) if section == "resources" else value)
    assert getattr(owner, other_field) == (NormalIntDistribution(other_value, 0) if section == "resources" else other_value)


@pytest.mark.parametrize("field,value", [
    ("save_with_layer", "false"), ("save_with_layer", 0), ("save_with_layer", 1),
    ("save_without_layer", "true"), ("save_without_layer", 0), ("save_without_layer", 1),
    ("base_seed", 1.5), ("base_seed", "7"), ("base_seed", True), ("base_seed", float("nan")),
])
def test_invalid_run_flags_and_seed_rejected(field, value):
    with pytest.raises(ValueError):
        RunConfig(generator=default_config(), **{field: value})


@pytest.mark.parametrize("section", ["topology", "resources"])
@pytest.mark.parametrize("nested_first", [False, True])
def test_nested_and_flat_distribution_fields_merge_without_mutation(section, nested_first):
    field = "out_degree" if section == "topology" else "cpu_time_ms"
    items = [(section, {field: {"mean": 3, "maximum": 9}}), (field, {"variance": 2})]
    overrides = dict(items if nested_first else reversed(items))
    snapshot = deepcopy(overrides)
    distribution = getattr(getattr(load_config(**overrides), section), field)
    assert distribution == NormalIntDistribution(3, 2, 0 if section == "topology" else 1, 9)
    assert overrides == snapshot
