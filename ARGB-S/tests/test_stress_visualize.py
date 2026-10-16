from collections import defaultdict
from html.parser import HTMLParser
import json
import math
import re
import xml.etree.ElementTree as ET

import pytest

from graph_generator import generate_graph, visualize
from .stress_support import expected_dataset, small_config


class HtmlInventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.panels = []
        self.options = []
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section":
            self.panels.append(attrs)
        elif tag == "option":
            self.options.append(attrs)
        elif tag == "script":
            self.scripts += 1
        assert not any(name.startswith("on") for name in attrs if name != "onchange")


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("cap", [0, 1, 2, 4, 16])
@pytest.mark.parametrize("selected", range(4))
def test_visualization_visibility_geometry_and_escaping(tmp_path, seed, cap, selected):
    graphs = [
        generate_graph(id=f'图_{seed}_{index}_"><script>alert(1)</script>&', seed=seed + index, config=small_config(seed + index))
        for index in range(4)
    ]
    dataset = expected_dataset(graphs)
    input_path = tmp_path / "数据.json"
    output_path = tmp_path / "预览" / "graph.html"
    input_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    visualize(input_path=input_path, output_path=output_path, graph_index=selected, max_nodes_per_layer=cap)
    document = output_path.read_text(encoding="utf-8")
    parsed = HtmlInventory()
    parsed.feed(document)
    assert parsed.scripts == 1
    assert len(parsed.panels) == len(parsed.options) == 4
    assert [int(option["value"]) for option in parsed.options if "selected" in option] == [selected]
    svgs = re.findall(r"<svg\b.*?</svg>", document, re.DOTALL)
    assert len(svgs) == 4
    for index, (graph, panel, svg_text) in enumerate(zip(graphs, parsed.panels, svgs)):
        assert panel["data-graph-id"] == graph.graph["id"]
        assert ("active" in panel["class"].split()) == (index == selected)
        assert panel["data-graph-index"] == str(index)
        grouped = defaultdict(list)
        for node in graph.nodes():
            grouped[node.layer].append(node)
        visible = [node for nodes in grouped.values() for node in nodes[:cap]]
        visible_ids = {node.node_id for node in visible}
        edges = [edge for edge in graph.edges() if edge.source in visible_ids and edge.target in visible_ids]
        svg = ET.fromstring(svg_text)
        assert svg.find("title").text == graph.graph["id"]
        width, height = float(svg.attrib["width"]), float(svg.attrib["height"])
        assert width > 0 and height > 0
        circles, lines = svg.findall("circle"), svg.findall("line")
        assert len(circles) == len(visible)
        assert len(lines) == len(edges)
        positions = {}
        for node, circle in zip(visible, circles):
            x, y = float(circle.attrib["cx"]), float(circle.attrib["cy"])
            assert math.isfinite(x) and math.isfinite(y)
            assert 0 < x < width and 0 < y < height
            assert circle.attrib["fill"] == ("#4f7cff" if node.node_type == "GPU" else "#ffb981")
            assert circle.find("title").text.startswith(node.node_id)
            positions[node.node_id] = (x, y)
        for edge, line in zip(edges, lines):
            assert (float(line.attrib["x1"]), float(line.attrib["y1"])) == positions[edge.source]
            assert (float(line.attrib["x2"]), float(line.attrib["y2"])) == positions[edge.target]
        assert f"hidden nodes: {len(graph.nodes()) - len(visible)}" in document
    assert json.loads(input_path.read_text(encoding="utf-8")) == dataset


@pytest.mark.parametrize("field,value", [
    ("graph_index", -1), ("graph_index", 1), ("graph_index", 0.5), ("graph_index", True), ("graph_index", "0"),
    ("max_nodes_per_layer", -1), ("max_nodes_per_layer", 1.5),
    ("max_nodes_per_layer", True), ("max_nodes_per_layer", "1"), ("max_nodes_per_layer", None),
])
def test_invalid_visualization_options_preserve_output(tmp_path, field, value):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.html"
    input_path.write_text('{"graphs": [{"id": "empty", "nodes": [], "edges": []}]}', encoding="utf-8")
    output_path.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError):
        visualize(input_path=input_path, output_path=output_path, **{field: value})
    assert output_path.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("field", ["total_tokens", "total_cpu_time_ms", "total_gpu_time_ms"])
def test_cost_summary_escapes_external_values(tmp_path, field):
    payload = '<script>alert("cost")</script>'
    data = {"graphs": [{"id": "cost", "nodes": [], "edges": [], "estimated_cost": {field: payload}}]}
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.html"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    visualize(input_path=input_path, output_path=output_path)
    document = output_path.read_text(encoding="utf-8")
    parsed = HtmlInventory()
    parsed.feed(document)
    assert parsed.scripts == 1
    assert payload not in document
    assert "&lt;script&gt;" in document
