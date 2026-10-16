"""Visualize generated graph JSON as a standalone HTML file."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


NODE_COLORS = {
    "GPU": "#4f7cff",
    "CPU": "#ffb981",
}

X_GAP = 180
Y_GAP = 28
NODE_RADIUS = 6
MARGIN_X = 80
MARGIN_Y = 80


def visualize(
    input_path: str | Path = "./outputs/generated_graphs.json",
    output_path: str | Path = "./outputs/graph_view.html",
    graph_index: int = 0,
    max_nodes_per_layer: int = 500,
) -> None:
    """Read graph JSON and generate a standalone HTML visualization."""
    if isinstance(graph_index, bool) or not isinstance(graph_index, int) or graph_index < 0:
        raise ValueError("graph_index must be a nonnegative integer")
    if (
        isinstance(max_nodes_per_layer, bool)
        or not isinstance(max_nodes_per_layer, int)
        or max_nodes_per_layer < 0
    ):
        raise ValueError("max_nodes_per_layer must be a nonnegative integer")
    data = _read_json(input_path)
    graph_items = data.get("graphs", [])

    if not graph_items:
        raise ValueError("JSON does not contain any graphs")
    if graph_index < 0 or graph_index >= len(graph_items):
        raise ValueError(f"graph_index is out of range: {graph_index}; graph count: {len(graph_items)}")

    html_text = _build_html(graph_items, graph_index, max_nodes_per_layer)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_html(
    graph_items: list[dict[str, Any]],
    selected_index: int,
    max_nodes_per_layer: int,
) -> str:
    options = _build_graph_options(graph_items, selected_index)
    panels = [
        _build_graph_panel(graph_data, index, len(graph_items), selected_index, max_nodes_per_layer)
        for index, graph_data in enumerate(graph_items)
    ]
    first_title = graph_items[selected_index].get("id", "unknown")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Graph View - {_escape(first_title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    header {{
      padding: 20px 28px;
      background: white;
      border-bottom: 1px solid #e5e7eb;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 20px;
      font-weight: 700;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
      font-size: 13px;
      color: #374151;
    }}
    select {{
      min-width: 220px;
      padding: 6px 8px;
      border: 1px solid #d1d5db;
      background: white;
      color: #111827;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      font-size: 13px;
      color: #4b5563;
      margin-bottom: 16px;
    }}
    .legend {{
      margin-top: 12px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .canvas {{
      overflow: auto;
      padding: 24px;
    }}
    .graph-panel {{
      display: none;
    }}
    .graph-panel.active {{
      display: block;
    }}
    svg {{
      background: white;
      border: 1px solid #e5e7eb;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    .edge {{
      stroke: #94a3b8;
      stroke-width: 1;
      stroke-opacity: 0.18;
    }}
    .node {{
      stroke: white;
      stroke-width: 1.5;
    }}
    .layer-label {{
      fill: #374151;
      font-size: 12px;
      font-weight: 700;
      text-anchor: middle;
    }}
    .hidden-label {{
      fill: #9ca3af;
      font-size: 11px;
      text-anchor: middle;
    }}
  </style>
</head>
<body>
  <header>
    <h1 id="page-title">{_escape(first_title)}</h1>
    <div class="toolbar">
      <label for="graph-select">Graph</label>
      <select id="graph-select" onchange="switchGraph(this.value)">
        {options}
      </select>
    </div>
    {_build_legend()}
  </header>
  <main class="canvas">
    {"".join(panels)}
  </main>
  <script>
    function switchGraph(index) {{
      const panels = document.querySelectorAll(".graph-panel");
      panels.forEach(panel => panel.classList.remove("active"));

      const active = document.querySelector(`[data-graph-index="${{index}}"]`);
      if (!active) {{
        return;
      }}

      active.classList.add("active");
      document.getElementById("page-title").textContent = active.dataset.graphId;
      document.getElementById("graph-select").value = index;
    }}

    switchGraph("{selected_index}");
  </script>
</body>
</html>
"""


def _build_graph_options(graph_items: list[dict[str, Any]], selected_index: int) -> str:
    options = []
    for index, graph_data in enumerate(graph_items):
        selected = " selected" if index == selected_index else ""
        label = f'{index}: {graph_data.get("id", "unknown")}'
        options.append(
            f'<option value="{index}"{selected}>{_escape(label)}</option>'
        )
    return "".join(options)


def _build_graph_panel(
    graph_data: dict[str, Any],
    graph_index: int,
    total_graphs: int,
    selected_index: int,
    max_nodes_per_layer: int,
) -> str:
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    layers = _group_nodes_by_layer(nodes)
    visible_nodes, hidden_counts = _pick_visible_nodes(layers, max_nodes_per_layer)
    positions, svg_width, svg_height = _compute_positions(visible_nodes)

    visible_node_ids = set(positions)
    visible_edges = [
        edge
        for edge in edges
        if edge["source"] in visible_node_ids and edge["target"] in visible_node_ids
    ]

    summary = _build_summary(
        graph_data=graph_data,
        graph_index=graph_index,
        total_graphs=total_graphs,
        total_nodes=len(nodes),
        total_edges=len(edges),
        visible_nodes=len(visible_node_ids),
        visible_edges=len(visible_edges),
        hidden_counts=hidden_counts,
    )
    svg = _build_svg(
        graph_data,
        visible_nodes,
        visible_edges,
        positions,
        svg_width,
        svg_height,
    )
    active_class = " active" if graph_index == selected_index else ""
    id = _escape(graph_data.get("id", "unknown"))

    return f"""
    <section class="graph-panel{active_class}" data-graph-index="{graph_index}" data-graph-id="{id}">
      {summary}
      {svg}
    </section>
    """


def _group_nodes_by_layer(nodes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    layers = {}
    for node in nodes:
        layer = int(node.get("layer", 0))
        layers.setdefault(layer, []).append(node)
    return dict(sorted(layers.items()))


def _pick_visible_nodes(
    layers: dict[int, list[dict[str, Any]]],
    max_nodes_per_layer: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    visible = {}
    hidden_counts = {}
    for layer, nodes in layers.items():
        visible[layer] = nodes[:max_nodes_per_layer]
        hidden_counts[layer] = max(0, len(nodes) - len(visible[layer]))
    return visible, hidden_counts


def _compute_positions(
    visible_nodes: dict[int, list[dict[str, Any]]],
) -> tuple[dict[str, tuple[float, float]], int, int]:
    layer_count = max(visible_nodes.keys(), default=0) + 1
    max_width = max((len(nodes) for nodes in visible_nodes.values()), default=1)

    svg_width = MARGIN_X * 2 + max(1, layer_count - 1) * X_GAP
    svg_height = MARGIN_Y * 2 + max(1, max_width - 1) * Y_GAP

    positions = {}
    for layer, nodes in visible_nodes.items():
        x = MARGIN_X + layer * X_GAP
        layer_height = max(1, len(nodes) - 1) * Y_GAP
        start_y = (svg_height - layer_height) / 2

        for index, node in enumerate(nodes):
            y = start_y + index * Y_GAP
            positions[node["node_id"]] = (x, y)

    return positions, svg_width, svg_height


def _build_summary(
    graph_data: dict[str, Any],
    graph_index: int,
    total_graphs: int,
    total_nodes: int,
    total_edges: int,
    visible_nodes: int,
    visible_edges: int,
    hidden_counts: dict[int, int],
) -> str:
    hidden_total = sum(hidden_counts.values())
    cost = graph_data.get("estimated_cost", {})
    total_tokens = cost.get("total_tokens", 0)
    total_cpu_time_ms = cost.get("total_cpu_time_ms", 0)
    total_gpu_time_ms = cost.get("total_gpu_time_ms", 0)

    return f"""
    <div class="meta">
      <span>Graph {graph_index + 1} / {total_graphs}</span>
      <span>nodes: {total_nodes} ({visible_nodes} visible)</span>
      <span>edges: {total_edges} ({visible_edges} visible)</span>
      <span>hidden nodes: {hidden_total}</span>
      <span>tokens: {_escape(total_tokens)}</span>
      <span>cpu ms: {_escape(total_cpu_time_ms)}</span>
      <span>gpu ms: {_escape(total_gpu_time_ms)}</span>
    </div>
    """


def _build_legend() -> str:
    items = []
    for node_type, color in NODE_COLORS.items():
        items.append(
            f'<span><i class="swatch" style="background:{color}"></i>'
            f'{_escape(node_type)}</span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


def _build_svg(
    graph_data: dict[str, Any],
    visible_nodes: dict[int, list[dict[str, Any]]],
    visible_edges: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> str:
    edge_svg = [_edge_to_svg(edge, positions) for edge in visible_edges]
    node_svg = [
        _node_to_svg(node, positions[node["node_id"]])
        for nodes in visible_nodes.values()
        for node in nodes
    ]
    label_svg = _layer_labels_to_svg(visible_nodes, height)

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <title>{_escape(graph_data.get("id", "unknown"))}</title>
      {"".join(edge_svg)}
      {"".join(node_svg)}
      {"".join(label_svg)}
    </svg>
    """


def _edge_to_svg(
    edge: dict[str, Any],
    positions: dict[str, tuple[float, float]],
) -> str:
    x1, y1 = positions[edge["source"]]
    x2, y2 = positions[edge["target"]]
    title = (
        f'{edge.get("edge_id", "")}: '
        f'{edge["source"]} -> {edge["target"]}'
    )
    return (
        f'<line class="edge" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
        f'<title>{_escape(title)}</title></line>'
    )


def _node_to_svg(node: dict[str, Any], position: tuple[float, float]) -> str:
    x, y = position
    node_type = node.get("node_type", "unknown")
    color = NODE_COLORS.get(node_type, "#64748b")
    title = (
        f'{node.get("node_id", "")}\\n'
        f'type: {node_type}\\n'
        f'layer: {node.get("layer", "")}'
    )
    if "time_ms" in node:
        title = f'{title}\\ntime_ms: {node["time_ms"]}'

    return (
        f'<circle class="node" cx="{x}" cy="{y}" r="{NODE_RADIUS}" '
        f'fill="{color}"><title>{_escape(title)}</title></circle>'
    )


def _layer_labels_to_svg(
    visible_nodes: dict[int, list[dict[str, Any]]],
    height: int,
) -> list[str]:
    labels = []
    for layer, nodes in visible_nodes.items():
        x = MARGIN_X + layer * X_GAP
        labels.append(
            f'<text class="layer-label" x="{x}" y="28">L{layer}</text>'
        )
        labels.append(
            f'<text class="hidden-label" x="{x}" y="{height - 24}">'
            f'{len(nodes)} shown</text>'
        )
    return labels


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)
