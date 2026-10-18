# StrategyQA

## Source and conversion

- Upstream source: StrategyQA training examples.
- Conversion: frozen full LLM rerun.
- Workload shape: yes/no multi-hop reasoning DAGs, commonly with parallel
  evidence or comparison branches followed by answer aggregation.

## Released files

| File | Role |
|---|---|
| `dags.json` | 2,061 DAGs (7,642 nodes and 6,090 edges) |
| `resources.json` | Matching calibrated resource sidecar |
