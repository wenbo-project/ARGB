from __future__ import annotations

from src.dag_standard import LOCAL_SIM_CPU_PROFILE, build_resource_estimates

def test_local_sim_profile_replaces_legacy_prompt_length_estimate() -> None:
    dag = _single_node_dag(
        {
            "node_id": "format",
            "node_type": "CPU",
            "prompt": "x" * 1000,
            "metadata": {"operation_type": "formatting"},
            "cpu": {"time_ms": 2100},
        }
    )

    legacy = build_resource_estimates([dag])
    calibrated = build_resource_estimates(
        [dag], cpu_estimation_profile=LOCAL_SIM_CPU_PROFILE
    )

    assert legacy["dags"][0]["nodes"][0]["cpu"]["time_ms"] == 2100
    cpu = calibrated["dags"][0]["nodes"][0]["cpu"]
    assert cpu == {
        "time_ms": 3,
        "source": "estimated",
        "profile": LOCAL_SIM_CPU_PROFILE,
        "operation_class": "formatting",
        "payload_chars": 1000,
    }


def test_tool_call_uses_arguments_instead_of_repeated_task_prompt() -> None:
    common = {
        "node_id": "tool",
        "node_type": "CPU",
        "metadata": {
            "action_name": "lookup",
            "arguments": {"id": "abc"},
        },
        "cpu": {"time_ms": 5000},
    }
    short = _single_node_dag({**common, "prompt": "call lookup"})
    long = _single_node_dag({**common, "prompt": "task context " * 1000})

    short_cpu = _calibrated_cpu(short)
    long_cpu = _calibrated_cpu(long)

    assert short_cpu["operation_class"] == "tool_call"
    assert short_cpu["payload_chars"] == long_cpu["payload_chars"]
    assert short_cpu["time_ms"] == long_cpu["time_ms"]
    assert short_cpu["time_ms"] < 100


def test_local_sim_profile_preserves_measured_cpu_time() -> None:
    dag = _single_node_dag(
        {
            "node_id": "measured",
            "node_type": "CPU",
            "prompt": "format result",
            "metadata": {"operation_type": "formatting"},
            "cpu": {"time_ms": 7, "source": "measured"},
        }
    )

    cpu = _calibrated_cpu(dag)

    assert cpu == {"time_ms": 7, "source": "measured"}


def test_compact_sidecar_omits_prompt_but_keeps_auditable_cpu_fields() -> None:
    dag = _single_node_dag(
        {
            "node_id": "validate",
            "node_type": "CPU",
            "prompt": "validate output",
            "metadata": {"operation_type": "validation"},
        }
    )

    payload = build_resource_estimates(
        [dag],
        cpu_estimation_profile=LOCAL_SIM_CPU_PROFILE,
        include_node_context=False,
    )
    node = payload["dags"][0]["nodes"][0]

    assert "prompt" not in node
    assert "metadata" not in node
    assert node["cpu"]["operation_class"] == "validation"
    assert payload["metadata"]["node_context_included"] is False
    assert payload["metadata"]["cpu_duration_rules"]["tool_call"]["base_ms"] == 15


def test_environment_actions_use_action_profile() -> None:
    dag = _single_node_dag(
        {
            "node_id": "act",
            "node_type": "CPU",
            "prompt": "Take the action PickupObject(alarmclock).",
            "metadata": {"original_node_type": "high_level_action"},
        }
    )

    cpu = _calibrated_cpu(dag)

    assert cpu["operation_class"] == "action"
    assert 20 <= cpu["time_ms"] <= 120


def _calibrated_cpu(dag: dict) -> dict:
    payload = build_resource_estimates(
        [dag], cpu_estimation_profile=LOCAL_SIM_CPU_PROFILE
    )
    return payload["dags"][0]["nodes"][0]["cpu"]


def _single_node_dag(node: dict) -> dict:
    return {
        "dag_id": "dag_1",
        "dag_type": "test",
        "nodes": [node],
        "edges": [],
    }
