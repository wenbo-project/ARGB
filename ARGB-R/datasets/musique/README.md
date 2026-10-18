# MuSiQue

## Source and conversion

- Upstream source: MuSiQue `train.jsonl`.
- Conversion: frozen full LLM conversion using the optimized prompt and rerun
  configuration.
- Workload shape: multi-hop question-answering DAGs with decomposition,
  retrieval/reasoning branches, and final aggregation.

## Released files

| File | Role |
|---|---|
| `dags.json` | 19,938 DAGs (95,044 nodes and 85,668 edges) |
| `resources.json` | Matching calibrated resource sidecar |
