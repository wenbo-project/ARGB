# ToolBench

## Source and conversion

- Upstream source: ToolBench `toolllama_G123_dfs_train.json`.
- Conversion: the frozen merged 26,999-graph DeepSeek LLM conversion used by
  the paper's benchmark table and real-data experiments.
- Workload shape: agent tool-use workflows; independent tool calls may appear
  as parallel branches.

## Released files

| File | Role |
| --- | --- |
| `dags-00001-of-00002.json` | First 13,500 graphs in original order |
| `dags-00002-of-00002.json` | Remaining 13,499 graphs |
| `dags.index.json` | Ordered shard paths, counts, and SHA-256 hashes |
| `resources.json` | Matching calibrated resource sidecar for both shards |

The two JSON arrays contain 26,999 graphs, 124,408 nodes, and 131,936 edges.
Read the index, resolve shard paths relative to its directory, and load the
arrays in the listed order. Keep the shared resource file with both shards.

Graph IDs, structure, node order, resource labels, and frozen numerical costs
are preserved. Historical schema identifiers and provenance metadata remain
unchanged. Partial checkpoints, the earlier 3,000-graph subset, and API/debug
samples are excluded.
