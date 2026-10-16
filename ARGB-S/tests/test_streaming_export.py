import json

import pytest

from graph_generator import (
    RunConfig,
    config_to_dict,
    generate_graphs,
    generate_graph,
    graph_to_dict,
    graphs_to_dataset,
    iter_graphs,
    load_config,
    save_dataset,
    save_dataset_no_layer,
    generate_and_save,
)


def small_config():
    return load_config(layer_count=3, out_degree=2)


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("with_config", [False, True])
@pytest.mark.parametrize("with_layer", [False, True])
def test_streaming_output_matches_dataset(tmp_path, count, with_config, with_layer):
    config = small_config()
    graphs = [
        generate_graph(id=f"g{i}", seed=i, config=config)
        for i in range(count)
    ]
    final_config = config if with_config else None
    expected = graphs_to_dataset(graphs, config=final_config)
    if not with_layer:
        for graph_data in expected["graphs"]:
            for node in graph_data["nodes"]:
                node.pop("layer")
    path = tmp_path / "nested" / "data.json"
    save = save_dataset if with_layer else save_dataset_no_layer
    save(iter(graphs), path, config=final_config)
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert all(node.layer >= 0 for graph in graphs for node in graph.nodes())


def test_each_graph_is_serialized_before_requesting_next(tmp_path, monkeypatch):
    import graph_generator.export as export

    config = small_config()
    serialized = []
    original_dump = export.json.dump

    def record_dump(data, writer, **kwargs):
        original_dump(data, writer, **kwargs)
        if "id" in data:
            serialized.append(data["id"])

    monkeypatch.setattr(export.json, "dump", record_dump)

    def graphs():
        for index in range(3):
            assert serialized == [f"g{i}" for i in range(index)]
            yield generate_graph(id=f"g{index}", seed=index, config=config)
        assert serialized == ["g0", "g1", "g2"]

    path = tmp_path / "stream.json"
    save_dataset(graphs(), path, config=config)
    assert len(json.loads(path.read_text(encoding="utf-8"))["graphs"]) == 3


@pytest.mark.parametrize("thread_count", [1, 3])
@pytest.mark.parametrize("count", [0, 1, 7])
def test_bounded_generation_preserves_seed_and_order(thread_count, count):
    config = small_config()
    run = RunConfig(
        generator=config, graph_count=count, thread_count=thread_count, base_seed=13,
    )
    expected = [
        graph_to_dict(generate_graph(
            id=f"{index:06d}", seed=13000 + index, config=config,
        ))
        for index in range(count)
    ]
    assert [graph_to_dict(graph) for graph in iter_graphs(run)] == expected
    assert [graph_to_dict(graph) for graph in generate_graphs(run)] == expected


def test_iterator_does_not_queue_whole_batch(monkeypatch):
    import graph_generator.export as export

    submitted = []
    consumed = []

    class Future:
        def __init__(self, index):
            self.index = index

        def result(self):
            return self.index

    class Executor:
        def __init__(self, max_workers):
            self.limit = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, index):
            submitted.append(index)
            assert len(submitted) - len(consumed) <= self.limit
            return Future(index)

    monkeypatch.setattr(export, "ThreadPoolExecutor", Executor)
    run = RunConfig(generator=small_config(), graph_count=20, thread_count=3)
    for graph in iter_graphs(run):
        consumed.append(graph)
    assert consumed == list(range(20))
    assert submitted == consumed


@pytest.mark.parametrize("retain_graphs", [False, True])
def test_batch_writes_both_variants_once(tmp_path, monkeypatch, retain_graphs):
    import graph_generator.export as export

    config = small_config()
    run = RunConfig(
        generator=config, graph_count=4, thread_count=2,
        output_path_with_layer=str(tmp_path / "with.json"),
        output_path_without_layer=str(tmp_path / "without.json"),
    )
    conversions = []
    original_convert = export.graph_to_dict

    def record_convert(graph):
        conversions.append(graph.graph["id"])
        return original_convert(graph)

    monkeypatch.setattr(export, "graph_to_dict", record_convert)
    result = generate_and_save(run, retain_graphs=retain_graphs)
    with_layer = json.loads((tmp_path / "with.json").read_text(encoding="utf-8"))
    without_layer = json.loads((tmp_path / "without.json").read_text(encoding="utf-8"))
    assert len(result) == (4 if retain_graphs else 0)
    assert conversions == [f"{i:06d}" for i in range(4)]
    assert with_layer["config"] == config_to_dict(config)
    assert all("layer" in node for graph_data in with_layer["graphs"] for node in graph_data["nodes"])
    assert all("layer" not in node for graph_data in without_layer["graphs"] for node in graph_data["nodes"])
    for graph_data in with_layer["graphs"]:
        for node in graph_data["nodes"]:
            node.pop("layer")
    assert with_layer == without_layer


@pytest.mark.parametrize("with_layer", [False, True])
def test_batch_single_output_and_empty_batch(tmp_path, with_layer):
    path = tmp_path / "single.json"
    run = RunConfig(
        generator=small_config(), graph_count=0,
        save_with_layer=with_layer, save_without_layer=not with_layer,
        output_path_with_layer=str(path), output_path_without_layer=str(path),
    )
    assert generate_and_save(run, retain_graphs=False) == []
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "graphs": [], "config": config_to_dict(run.generator),
    }


def test_duplicate_output_paths_raise_before_writing(tmp_path):
    path = tmp_path / "same.json"
    path.write_text("existing", encoding="utf-8")
    run = RunConfig(
        generator=small_config(),
        output_path_with_layer=str(path),
        output_path_without_layer=str(path),
    )
    with pytest.raises(ValueError, match="output paths must be distinct"):
        generate_and_save(run)
    assert path.read_text(encoding="utf-8") == "existing"
