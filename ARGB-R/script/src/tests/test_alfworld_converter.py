from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


from src import _alfworld as converter


def _candidate(index: int) -> dict[str, int]:
    return {"ordered_index": index}


def _dag(source_index: int | str) -> dict[str, object]:
    return {"metadata": {"source_index": source_index}}


def test_pending_candidates_retries_gaps_before_new_sources() -> None:
    ordered = [_candidate(index) for index in range(6)]
    existing = [_dag(1), _dag("3")]

    pending = converter.pending_candidates(ordered, existing, start=0, limit=0)

    assert [candidate["ordered_index"] for candidate in pending] == [0, 2, 4, 5]


def test_pending_candidates_limit_is_final_output_size() -> None:
    ordered = [_candidate(index) for index in range(6)]
    existing = [_dag(1), _dag(3)]

    pending = converter.pending_candidates(ordered, existing, start=0, limit=3)

    assert [candidate["ordered_index"] for candidate in pending] == [0]


def test_pending_candidates_rejects_duplicate_source_indexes() -> None:
    with pytest.raises(ValueError, match="duplicate metadata.source_index"):
        converter.pending_candidates([_candidate(0)], [_dag(0), _dag("0")], start=0, limit=0)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 405, 413, 422])
def test_non_retryable_http_statuses_fail_fast(status: int) -> None:
    error = RuntimeError(f"HTTP Error {status}: provider rejected the request")

    assert converter.llm_http_status(error) == status
    assert converter.is_non_retryable_llm_error(error)


def test_rate_limit_remains_retryable() -> None:
    error = RuntimeError("HTTP Error 429: Too Many Requests")

    assert converter.llm_http_status(error) == 429
    assert not converter.is_non_retryable_llm_error(error)


def test_billing_error_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "traj_data.json"
    source.write_text("{}", encoding="utf-8")
    calls = 0

    def reject_request(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("HTTP Error 402: Payment Required")

    monkeypatch.setattr(converter, "call_llm", reject_request)
    args = SimpleNamespace(llm_retries=2, skip_invalid_llm=True)
    candidate = {
        "path": source,
        "ordered_index": 7,
        "split": "train",
        "task_type": "pick_and_place_simple",
        "task_instance": "example",
    }

    with pytest.raises(converter.FatalLLMError, match="source index 7"):
        converter.convert_one_llm(candidate, 0, args)

    assert calls == 1


def test_invalid_top_level_metadata_is_a_retryable_validation_error() -> None:
    dag = {
        "dag_id": "alfworld_0001",
        "original_question": "Move the object.",
        "nodes": [
            {"node_id": "N0", "node_type": "GPU", "prompt": "Plan."},
            {"node_id": "N1", "node_type": "GPU", "prompt": "Locate."},
            {"node_id": "N2", "node_type": "CPU", "prompt": "Move."},
            {"node_id": "N3", "node_type": "GPU", "prompt": "Validate."},
        ],
        "edges": [
            {"source": "N0", "target": "N1"},
            {"source": "N1", "target": "N2"},
            {"source": "N2", "target": "N3"},
        ],
        "metadata": "llm",
    }

    with pytest.raises(ValueError, match="metadata must be a JSON object"):
        converter.validate_llm_dag(dag)
