# Configuration guide

This guide explains how configuration values become graph properties and how the controls interact. For installation and basic usage, see [README.md](../README.md); for output and API details, see the [technical reference](reference.md).

## Loading and overrides

The packaged [default.toml](../graph_generator/configs/default.toml) is the source of generator defaults. Settings are merged in the following order, with later values taking precedence:

```text
default.toml → TOML specified by config_path → Python keyword overrides
```

User files need a `[generator]` section but may omit individual fields. Omitted fields are inherited, including individual fields within distribution tables.

```toml
[generator.topology]
out_degree = { mean = 3 }
max_layer_jump = 2
max_layer_width = 50
```

```python
from graph_generator import generate_graph, load_config

config = load_config("my_config.toml", max_total_nodes=500)
graph = generate_graph(id="custom", config=config, seed=7)
```

`generate_graph(config=...)` accepts a finished configuration and cannot also receive `config_path` or generator keyword overrides. Build the merged configuration with `load_config()` first. `id` and `seed` remain separate arguments.

Unknown fields raise errors. Packaged defaults are loaded independently of the current working directory; relative user file paths are resolved from the current working directory. Run-level settings such as `graph_count` and output paths belong to `RunConfig`.

## Distribution values become bounded integers

Layer count, entry width, out-degree, execution times, and GPU token counts use the same integer sampling rule:

```text
X ~ Normal(mean, variance)
value = max(minimum, round(X))
if maximum is set: value = min(value, maximum)
```

The standard deviation is `sqrt(variance)`. With `variance=0`, the value is fixed after rounding and clamping. The implementation uses Python's `round`, including its ties-to-even rule. Samples outside the bounds are clamped to the boundaries without resampling.

Consequently, `mean` describes the underlying normal distribution; it is not necessarily the observed mean after rounding and clamping. This matters especially for out-degree distributions that allow zero.

Two override forms have different meanings:

| Override | Result |
|---|---|
| `out_degree={"mean": 4}` | Change the mean and inherit variance and bounds from the merged TOML configuration |
| `out_degree=4` | Replace the distribution with a fixed value: mean 4, variance 0, minimum 0, no maximum |

Scalar overrides of other distributions likewise create fixed distributions, with a minimum of 2 for layer count and 1 for entry width and resources. They replace previously configured distribution bounds.

## Parameter reference

### Node types

`[generator].node_types` defaults to `["GPU", "CPU"]`. Each node type is chosen from this pool. Only `CPU` and `GPU` are accepted, without duplicates; the pool must be non-empty. Type names are case-sensitive and cannot contain surrounding whitespace.

`[generator].gpu_ratio` defaults to `0.5` and must be a number between `0` and `1`. When both node types are enabled, each node is selected as GPU with probability `gpu_ratio`, and CPU with probability `1 - gpu_ratio`, regardless of the order in `node_types`. A ratio of `0` selects only CPU nodes; a ratio of `1` selects only GPU nodes. When the pool contains a single type, every node uses that type regardless of the ratio.

```toml
[generator]
node_types = ["GPU", "CPU"]
gpu_ratio = 0.3
```

This configuration gives each node a 30% probability of being GPU and a 70% probability of being CPU. The ratio is a sampling probability, so an individual graph is not guaranteed to contain exactly those proportions. In Python, use `load_config(gpu_ratio=0.3)` to override it.

### Graph shape: `[generator.topology]`

Distribution defaults below are written as `(mean, variance, minimum, maximum)`.

| Field | Default | Meaning |
|---|---|---|
| `layer_count` | `(40, 0, 2, 200)` | Requested total number of layers, including entry and final layers |
| `out_degree` | `(0.7, 16, 0, 10)` | Number of pending edge slots sampled for each node, before edge assignment and successor patching |
| `first_layer_width` | `(1, 0, 1, 50)` | Number of entry nodes, sampled once per graph |
| `max_layer_jump` | `1` | Maximum target/source layer difference for sampled edges |
| `max_layer_width` | `100` | Width cap for regular layers after the entry layer |
| `max_total_nodes` | `100000` | Strict total node budget, including entry and converge nodes |
| `force_converge` | `true` | Reserve one node and end the graph with a single sink |

The entry width uses its own distribution and is not clamped by `max_layer_width`. To bound it, set `first_layer_width.maximum`.

### Resource loads: `[generator.resources]`

| Field | Default `(mean, variance, minimum)` | Meaning |
|---|---|---|
| `cpu_time_ms` | `(1, 0, 1)` | CPU task execution time in milliseconds |
| `gpu_time_ms` | `(1, 0, 1)` | GPU task execution time in milliseconds |
| `gpu_token_count` | `(1, 0, 1)` | GPU task token count |

Resource distributions have no default maximum. GPU time and token count are sampled separately; there is no formula converting between them. These fields model task loads.

## How the topology controls interact

### Out-degree and jumps create future demand

Each emitting node samples an out-degree `D`, producing `D` edge slots that represent pending connection requests. For each slot, the generator samples a jump distance `J` and records the parent for target layer `source_layer + J`.

For `m = max_layer_jump`, raw weights decrease linearly as `[m, m-1, ..., 1]`:

```text
P(J = j) = (m - j + 1) / (m * (m + 1) / 2),  1 <= j <= m
```

| `max_layer_jump` | Probabilities for jumps 1, 2, 3 |
|---|---|
| 1 | `1` |
| 2 | `2/3, 1/3` |
| 3 | `1/2, 1/3, 1/6` |

A jump of 1 connects adjacent layers; a jump of 3 bypasses two intermediate layers. Slots targeting beyond the requested last layer are dropped, without resampling. Increasing the jump limit distributes demand across more future layers and can discard more slots near the tail.

### Pending slots, width cap, and budget determine layer width

For a regular layer after the entry layer, let `S` be its pending slot count, `Wmax` the width cap, and `R` the remaining regular-node budget. When `R >= 1`:

```text
layer_width = min(max(1, S), Wmax, R)
```

When `R < 1`, regular layer creation stops. The configured final layer uses one node when convergence is enabled.

If width and budget allow, each slot initially maps to a distinct child. If they limit the width, remaining slots choose among existing children, allowing connection requests to share child nodes. Duplicate parent/child connections are discarded. If `S=0`, the generator still creates one child and connects it to an eligible earlier node.

Thus, `out_degree`, jump distance, width cap, and total budget jointly determine the shape. The width cap limits the maximum width. A larger out-degree mean may increase pending demand, but it does not guarantee a particular width or node count.

### Sampled out-degree differs from final out-degree

After all layers are created, every non-terminal node without successors receives an edge to an existing node in the immediately next layer. This adds edges without creating nodes or changing widths. Terminal nodes receive no successor patching.

For example, with adjacent-layer edges and sufficient budget, five parents sampling `[2, 0, 1, 0, 1]` create four children. The two zero-out-degree parents are then connected to those children; the next layer stays four nodes wide. If all parents sample zero, the next layer stays one node wide.

Discarded tail slots, duplicate-edge removal, width limits, convergence, and successor patching can all make final out-degree differ from its sample. Non-terminal nodes sampled with zero out-degree obtain successors through patching.

### Convergence shares the total node budget

```text
regular_node_budget = max_total_nodes - int(force_converge)
```

With `force_converge=true`, entry and intermediate nodes share a budget of `max_total_nodes - 1`. The reserved last slot becomes a converge node. If regular nodes exhaust their budget before the requested final layer, a converge node is added immediately after the actual last regular layer.

With `force_converge=false`, regular nodes may consume the full budget and the last layer may contain multiple sinks. In both modes, budget exhaustion can reduce the actual number of layers below the sampled `layer_count`.

### Practical tuning

| Goal | Controls to consider | Interaction to check |
|---|---|---|
| Deeper graphs | Increase `layer_count` | Ensure the total budget supports the desired depth and widths |
| Wider or more branching graphs | Increase sampled `out_degree` or entry width | At the width or total-node limits, additional demand may share existing child nodes |
| More cross-layer dependencies | Increase `max_layer_jump` | Demand spreads across layers; tail slots may be dropped |
| Bounded datasets | Set `max_total_nodes` and `max_layer_width` | Entry width needs its own bound; convergence consumes one budget slot |
| One final sink | Enable `force_converge` | The reserved slot also supports early convergence |
| A regular tree | Use `tree.toml` | Fixed positive out-degree, adjacent-layer edges, ample width and budget, and disabled forced convergence preserve the tree structure |

For controlled comparisons, save the merged configuration and record the seed, and inspect the resulting graph statistics. Distribution means cannot be treated as exact structural measurements.

## Validation constraints

- Layer count has a minimum of at least 2; entry width and resource distributions require positive means and minimums of at least 1.
- Out-degree allows a mean and minimum of 0. Variance must be non-negative, and any maximum must be at least the minimum.
- `max_layer_jump` and `max_layer_width` must be at least 1.
- The total budget must fit the sampled entry width plus at least one subsequent node. Loading checks the minimum entry width and, when convergence is enabled, the fixed entry width. Sampled entry widths are checked again during generation.
