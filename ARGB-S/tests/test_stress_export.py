import json

import pytest

from graph_generator import (
    RunConfig, generate_and_save, generate_graph, graph_to_dict, graphs_to_dataset,
    save_dataset, save_dataset_no_layer,
)
from .stress_support import OnePass, expected_dataset, small_config


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("count", [0, 1, 4])
@pytest.mark.parametrize("include_config", [False, True])
@pytest.mark.parametrize("with_layer", [False, True])
def test_streamed_dataset_matches_independent_schema(tmp_path, seed, count, include_config, with_layer):
    config = small_config(seed)
    graphs = [generate_graph(id=f"图_{index}_é", seed=seed + index, config=config) for index in range(count)]
    snapshot = [graph_to_dict(graph) for graph in graphs]
    exported_config = config if include_config else None
    expected = expected_dataset(graphs, exported_config, with_layer)
    path = tmp_path / "输出" / "nested" / "数据.json"
    stream = OnePass(graphs)
    writer = save_dataset if with_layer else save_dataset_no_layer
    writer(stream, path, config=exported_config)
    assert stream.iterations == 1
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert [graph_to_dict(graph) for graph in graphs] == snapshot
    assert graphs_to_dataset(OnePass(graphs), config=exported_config) == expected_dataset(graphs, exported_config)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("mode", ["with", "without", "both"])
def test_batch_outputs_order_seed_and_retention(tmp_path, seed, count, threads, retain, mode):
    config = small_config(seed)
    with_path = tmp_path / "with.json"
    without_path = tmp_path / "without.json"
    for path in (with_path, without_path):
        path.write_text("preserve", encoding="utf-8")
    run = RunConfig(
        generator=config, graph_count=count, thread_count=threads, base_seed=seed,
        output_path_with_layer=str(with_path), output_path_without_layer=str(without_path),
        save_with_layer=mode != "without", save_without_layer=mode != "with",
    )
    expected = [generate_graph(id=f"{index:06d}", seed=seed * 1000 + index, config=config) for index in range(count)]
    returned = generate_and_save(run, retain_graphs=retain)
    assert [graph_to_dict(graph) for graph in returned] == ([graph_to_dict(graph) for graph in expected] if retain else [])
    for path, enabled, with_layer in ((with_path, run.save_with_layer, True), (without_path, run.save_without_layer, False)):
        text = path.read_text(encoding="utf-8")
        if enabled:
            assert json.loads(text) == expected_dataset(expected, config, with_layer)
        else:
            assert text == "preserve"
