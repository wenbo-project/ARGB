"""Directed graph container used by the graph generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    """Executable node in a graph."""

    node_id: str
    node_type: str
    layer: int
    tokens: dict[str, int] = field(default_factory=dict)
    time_ms: int | None = None


@dataclass(frozen=True)
class Edge:
    """Directed dependency edge in a graph."""

    edge_id: str
    source: str
    target: str


class DirectedGraph:
    """Directed graph container with node layer indexes."""

    def __init__(
        self,
        id: str,
        estimated_cost: dict[str, Any] | None = None,
    ) -> None:
        self.graph: dict[str, Any] = {
            "id": id,
            "estimated_cost": {} if estimated_cost is None else estimated_cost,
        }
        
        self._node: dict[str, Node] = {}
        self._node_in_layer: dict[int, list[str]] = {}

        self._edge: dict[str, Edge] = {}
        # Map each source node id to its outgoing edges, keyed by target node id.
        self._edges_from: dict[str, dict[str, Edge]] = {}
        # Map each target node id to its incoming edges, keyed by source node id.
        self._edges_to: dict[str, dict[str, Edge]] = {}

        

    def add_node(self, node: Node) -> None:
        """Add a node and initialize its adjacency indexes."""
        if node.node_id in self._node:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        if isinstance(node.layer, bool) or not isinstance(node.layer, int) or node.layer < 0:
            raise ValueError(f"invalid layer: {node.layer}")

        self._node[node.node_id] = node
        self._edges_from[node.node_id] = {}
        self._edges_to[node.node_id] = {}
        if node.layer not in self._node_in_layer:
            self._node_in_layer[node.layer] = []
        self._node_in_layer[node.layer].append(node.node_id)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge and keep all edge indexes in sync."""
        if edge.edge_id in self._edge:
            raise ValueError(f"duplicate edge_id: {edge.edge_id}")
        if edge.source not in self._node:
            raise ValueError(f"source does not exist: {edge.source}")
        if edge.target not in self._node:
            raise ValueError(f"target does not exist: {edge.target}")
        if edge.target in self._edges_from[edge.source]:
            raise ValueError(f"duplicate dependency edge: {edge.source} -> {edge.target}")

        self._edge[edge.edge_id] = edge
        self._edges_from[edge.source][edge.target] = edge
        self._edges_to[edge.target][edge.source] = edge

    def nodes(self) -> list[Node]:
        """Return all node objects."""
        return list(self._node.values())

    def edges(self) -> list[Edge]:
        """Return all edge objects."""
        return list(self._edge.values())

    def successors(self, node_id: str) -> list[str]:
        """Return all successor node ids for a node."""
        return list(self._edges_from[node_id])

    def predecessors(self, node_id: str) -> list[str]:
        """Return all predecessor node ids for a node."""
        return list(self._edges_to[node_id])

    def in_degree(self, node_id: str) -> int:
        """Return the in-degree for a node."""
        return len(self._edges_to[node_id])

    def out_degree(self, node_id: str) -> int:
        """Return the out-degree for a node."""
        return len(self._edges_from[node_id])

    def nodes_in_layer(self, layer: int) -> list[Node]:
        """Return all node objects in a layer."""
        return [
            self._node[node_id]
            for node_id in self._node_in_layer.get(layer, [])
        ]
