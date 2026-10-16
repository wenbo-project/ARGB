"""Configuration for graph generation runs.

One module for both halves of configuration: the model definitions
(dataclasses + validation) and the TOML loader/merger that builds them.

The packaged ``configs/default.toml`` is the single source of defaults, 
and ``load_config`` builds and validates a ``GeneratorConfig`` 
from TOML sources plus function overrides.

Precedence (lowest to highest):

1. packaged ``configs/default.toml`` (single source of defaults)
2. user config file passed as ``config_path``
3. function overrides passed as keyword arguments

The packaged default file is read via ``importlib.resources`` so loading
never depends on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

try:  # Python >= 3.11 ships tomllib
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from importlib import resources


def _check_at_least(name: str, value: int | float, minimum: int) -> None:
    """Require ``value >= minimum``, raising a descriptive error otherwise."""
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}: {value}")


def _check_integer(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer: {value!r}")
    _check_at_least(name, value, minimum)


def _check_number(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise ValueError(f"{name} must be a finite number: {value!r}")


@dataclass(frozen=True)
class NormalIntDistribution:
    """Integer values sampled from a normal distribution."""

    mean: float
    variance: float
    minimum: int = 1
    maximum: int | None = None

    def validate(self, name: str, *, allow_zero: bool = False) -> None:
        """Validate normal distribution settings."""
        _check_number(f"{name}.mean", self.mean)
        _check_number(f"{name}.variance", self.variance)
        if allow_zero:
            _check_at_least(f"{name}.mean", self.mean, 0)
        elif self.mean <= 0:
            raise ValueError(f"{name}.mean must be greater than 0: {self.mean}")
        if self.variance < 0:
            raise ValueError(f"{name}.variance cannot be negative: {self.variance}")
        _check_integer(f"{name}.minimum", self.minimum, 0 if allow_zero else 1)
        if self.maximum is not None:
            _check_integer(f"{name}.maximum", self.maximum, self.minimum)


@dataclass(frozen=True)
class TopologyConfig:
    """Parameters that control the generated graph shape."""

    layer_count: NormalIntDistribution
    out_degree: NormalIntDistribution
    max_layer_jump: int
    first_layer_width: NormalIntDistribution
    max_layer_width: int
    max_total_nodes: int
    force_converge: bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.layer_count.validate("topology.layer_count")
        _check_at_least("topology.layer_count.minimum", self.layer_count.minimum, 2)
        self.out_degree.validate("topology.out_degree", allow_zero=True)
        _check_integer("topology.max_layer_jump", self.max_layer_jump, 1)
        self.first_layer_width.validate("topology.first_layer_width")
        _check_integer("topology.max_layer_width", self.max_layer_width, 1)
        _check_integer("topology.max_total_nodes", self.max_total_nodes, 2)
        if not isinstance(self.force_converge, bool):
            raise ValueError("topology.force_converge must be a boolean")
        if self.max_total_nodes < self.first_layer_width.minimum + 1:
            raise ValueError(
                "topology.max_total_nodes must be at least first_layer_width.minimum + 1: "
                f"{self.max_total_nodes}"
            )
        if self.force_converge and self.first_layer_width.variance == 0:
            # A fixed entry width is known before generation; apply the same
            # rounding and bounds used by the sampler.
            width = max(
                self.first_layer_width.minimum, round(self.first_layer_width.mean)
            )
            if self.first_layer_width.maximum is not None:
                width = min(width, self.first_layer_width.maximum)
            if self.max_total_nodes < width + 1:
                raise ValueError(
                    "topology.max_total_nodes must reserve one converge node "
                    f"after fixed first_layer_width (requires {width + 1}): "
                    f"{self.max_total_nodes}"
                )


@dataclass(frozen=True)
class ResourceConfig:
    """Parameters that control generated resource and data costs."""

    cpu_time_ms: NormalIntDistribution
    gpu_time_ms: NormalIntDistribution
    gpu_token_count: NormalIntDistribution

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.cpu_time_ms.validate("resources.cpu_time_ms")
        self.gpu_time_ms.validate("resources.gpu_time_ms")
        self.gpu_token_count.validate("resources.gpu_token_count")


@dataclass(frozen=True)
class GeneratorConfig:
    """Tunable parameters for one generated graph.

    No code-level defaults: the packaged ``configs/default.toml`` is the
    single source of defaults. Build instances via ``load_config``.
    """

    node_types: tuple[str, ...]
    gpu_ratio: float
    topology: TopologyConfig
    resources: ResourceConfig

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.node_types:
            raise ValueError("node_types must not be empty")
        if not isinstance(self.node_types, (list, tuple)) or any(
            not isinstance(node_type, str) or node_type not in ("CPU", "GPU")
            for node_type in self.node_types
        ):
            raise ValueError("node_types must contain only CPU and GPU")
        if len(set(self.node_types)) != len(self.node_types):
            raise ValueError("node_types must not contain duplicate values")
        if (
            isinstance(self.gpu_ratio, bool)
            or not isinstance(self.gpu_ratio, (int, float))
            or not 0 <= self.gpu_ratio <= 1
        ):
            raise ValueError("gpu_ratio must be a number between 0 and 1")
        self.topology.validate()
        self.resources.validate()

    def validate_layer_count(self, layer_count: int) -> None:
        if layer_count < 2:
            raise ValueError(
                "layer_count must be at least 2 so the graph can contain "
                f"at least one parent-child layer transition: {layer_count}"
            )


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a batch generation run.

    Run-level parameters keep code defaults; ``generator`` has no default and
    must be built via ``load_config`` (single source of defaults).
    """

    graph_count: int = 10
    thread_count: int = 4
    output_path_with_layer: str = "./outputs/generated_graphs.json"
    output_path_without_layer: str = "./outputs/generated_graphs_no_layer.json"
    save_with_layer: bool = True
    save_without_layer: bool = True
    base_seed: int | None = 7
    generator: GeneratorConfig | None = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_integer("graph_count", self.graph_count, 0)
        _check_integer("thread_count", self.thread_count, 1)
        for name in ("save_with_layer", "save_without_layer"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.base_seed is not None and (
            isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int)
        ):
            raise ValueError("base_seed must be an integer or None")
        if not self.save_with_layer and not self.save_without_layer:
            raise ValueError("at least one output format must be enabled")
        if self.generator is None:
            raise ValueError(
                "RunConfig.generator is required; build it via load_config()"
            )
        self.generator.validate()


# ---------------------------------------------------------------------------
# Loading: TOML -> merged dict -> validated dataclasses
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> GeneratorConfig:
    """Build a validated ``GeneratorConfig`` from TOML sources and overrides."""
    defaults = _read_packaged_default()
    _validate_fields(defaults, frozenset({"generator"}), "config")
    values = defaults["generator"]
    _validate_generator_values(values)

    if config_path is not None:
        document = _read_toml(config_path)
        user = document.get("generator")
        if user is None:
            raise ValueError(
                f"missing [generator] section in config file: {config_path}"
            )
        _validate_fields(document, frozenset({"generator"}), "config")
        _validate_generator_values(user)
        values = _deep_merge(values, user)

    _validate_fields(overrides, _OVERRIDE_KEYS, "overrides")
    nested = _nest_overrides({
        key: value for key, value in overrides.items() if value is not None
    })
    _validate_generator_values(nested)
    values = _deep_merge(values, nested)

    config = _build_generator_config(values)
    config.validate()  # validate immediately after loading (design doc 11.3)
    return config


def _read_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def get_config_path(name: str = "default.toml") -> Path:
    """Absolute path of a config file shipped inside the package.

    Works regardless of the current working directory; handy for passing a
    packaged preset (e.g. ``large_graph.toml``) to ``load_config``.
    """
    res = resources.files("graph_generator").joinpath("configs").joinpath(name)
    if not res.is_file():
        raise FileNotFoundError(f"no packaged config file: configs/{name}")
    with resources.as_file(res) as path:
        return path


def _read_packaged_default() -> dict[str, Any]:
    data = (
        resources.files("graph_generator")
        .joinpath("configs/default.toml")
        .read_bytes()
    )
    return tomllib.loads(data.decode("utf-8"))


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Flat override keys mapped to their nested configuration paths.
_TOPOLOGY_KEYS = frozenset(
    {
        "layer_count",
        "out_degree",
        "max_layer_jump",
        "first_layer_width",
        "max_layer_width",
        "max_total_nodes",
        "force_converge",
    }
)
_RESOURCE_KEYS = frozenset({"cpu_time_ms", "gpu_time_ms", "gpu_token_count"})
_GENERATOR_KEYS = frozenset({"node_types", "gpu_ratio", "topology", "resources"})
_OVERRIDE_KEYS = _GENERATOR_KEYS | _TOPOLOGY_KEYS | _RESOURCE_KEYS
_DISTRIBUTION_KEYS = frozenset({"mean", "variance", "minimum", "maximum"})
_TOPOLOGY_DISTRIBUTIONS = frozenset({
    "layer_count", "out_degree", "first_layer_width"
})


def _validate_fields(
    values: dict[str, Any], allowed: frozenset[str], name: str,
) -> None:
    """Reject unknown keys before merging can hide invalid configuration."""
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a table/dict")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"unknown fields in {name}: {', '.join(sorted(unknown))}; "
            f"allowed fields: {', '.join(sorted(allowed))}"
        )


def _validate_generator_values(values: dict[str, Any]) -> None:
    """Check field names in generator sections and distribution tables."""
    _validate_fields(values, _GENERATOR_KEYS, "generator")
    for section, allowed, distributions in (
        ("topology", _TOPOLOGY_KEYS, _TOPOLOGY_DISTRIBUTIONS),
        ("resources", _RESOURCE_KEYS, _RESOURCE_KEYS),
    ):
        if section not in values:
            continue
        settings = values[section]
        _validate_fields(settings, allowed, f"generator.{section}")
        for name in distributions:
            if isinstance(settings.get(name), dict):
                _validate_fields(
                    settings[name], _DISTRIBUTION_KEYS,
                    f"generator.{section}.{name}",
                )


def _nest_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Map flat keyword overrides onto the nested generator structure.

    ``out_degree=3`` -> ``{"topology": {"out_degree": 3}}``
    ``cpu_time_ms=5``  -> ``{"resources": {"cpu_time_ms": 5}}``
    Nested dicts (e.g. ``topology={...}``, ``layer_count={...}``) are passed
    through unchanged so ``_deep_merge`` can recurse into them.
    """
    nested: dict[str, Any] = {
        key: dict(value) if key in ("topology", "resources") and isinstance(value, dict) else value
        for key, value in overrides.items()
        if key not in _TOPOLOGY_KEYS | _RESOURCE_KEYS
    }
    for key, value in overrides.items():
        if key in _TOPOLOGY_KEYS:
            section = "topology"
        elif key in _RESOURCE_KEYS:
            section = "resources"
        else:
            continue
        settings = nested.setdefault(section, {})
        if not isinstance(settings, dict):
            raise ValueError(f"generator.{section} must be a table/dict")
        nested[section] = _deep_merge(settings, {key: value})
    return nested


def _build_generator_config(values: dict[str, Any]) -> GeneratorConfig:
    if not isinstance(values["node_types"], (list, tuple)):
        raise ValueError("node_types must be a non-empty list/tuple of CPU and GPU")
    return GeneratorConfig(
        node_types=tuple(values["node_types"]),
        gpu_ratio=values["gpu_ratio"],
        topology=_build_topology(values["topology"]),
        resources=_build_resources(values["resources"]),
    )


def _build_topology(values: dict[str, Any]) -> TopologyConfig:
    return TopologyConfig(
        layer_count=_build_distribution(
            values["layer_count"], name="topology.layer_count", default_minimum=2
        ),
        out_degree=_build_distribution(
            values["out_degree"], name="topology.out_degree", default_minimum=0
        ),
        max_layer_jump=values["max_layer_jump"],
        first_layer_width=_build_distribution(
            values["first_layer_width"], name="topology.first_layer_width", default_minimum=1
        ),
        max_layer_width=values["max_layer_width"],
        max_total_nodes=values["max_total_nodes"],
        force_converge=values["force_converge"],
    )


def _build_resources(values: dict[str, Any]) -> ResourceConfig:
    return ResourceConfig(
        **{
            name: _build_distribution(
                values[name], name=f"resources.{name}", default_minimum=1
            )
            for name in ("cpu_time_ms", "gpu_time_ms", "gpu_token_count")
        }
    )


def _build_distribution(
    value: Any,
    name: str,
    default_minimum: int,
) -> NormalIntDistribution:
    """Normalize a scalar / dict / distribution object into a distribution."""
    if isinstance(value, NormalIntDistribution):
        return value
    if isinstance(value, dict):
        return NormalIntDistribution(
            mean=value["mean"],
            variance=value["variance"],
            minimum=value.get("minimum", default_minimum),
            maximum=value.get("maximum"),
        )
    if isinstance(value, (int, float)):
        # Scalar override -> fixed value (variance=0), minimum from context.
        return NormalIntDistribution(
            mean=value, variance=0.0, minimum=default_minimum
        )
    raise TypeError(
        f"{name} must be a number, dict, or NormalIntDistribution: {value!r}"
    )
