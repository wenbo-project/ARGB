# ALFWorld

## Source and conversion

- Upstream source: ALFWorld `json_2.1.1/train` trajectories.
- Conversion: OpenAI-compatible LLM conversion with DeepSeek.
- Selection: the frozen 6,374-DAG full LLM conversion used by the paper's
  benchmark table and real-data experiments.

## Released files

| File | Role |
|---|---|
| `dags.json` | 6,374 DAGs (76,455 nodes and 79,065 edges) |
| `resources.json` | Matching calibrated resource sidecar |

The earlier balanced 100-DAG sample is not part of this paper-aligned release.
