"""Synthetic layered graph generation package."""

from .config import (
    GeneratorConfig,
    NormalIntDistribution,
    ResourceConfig,
    RunConfig,
    TopologyConfig,
    load_config,
    get_config_path,
)
from .export import (
    config_to_dict,
    generate_graphs,
    graph_to_dict,
    graphs_to_dataset,
    iter_graphs,
    save_dataset,
    save_dataset_no_layer,
    generate_and_save,
)
from .generator import generate_graph
from .graph_structure import Edge, DirectedGraph, Node
from .visualize import visualize

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Edge",
    "DirectedGraph",
    "Node",
    "GeneratorConfig",
    "NormalIntDistribution",
    "ResourceConfig",
    "RunConfig",
    "TopologyConfig",
    "config_to_dict",
    "generate_graphs",
    "generate_graph",
    "graph_to_dict",
    "graphs_to_dataset",
    "iter_graphs",
    "load_config",
    "get_config_path",
    "save_dataset",
    "save_dataset_no_layer",
    "visualize",
    "generate_and_save",
]
