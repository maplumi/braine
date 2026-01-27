# Metrics (simple, first-principles)

We need measurements that:
- are **cheap** to compute on-device
- don’t require large datasets
- track the behaviors we care about (learn instantly, recall, habit, forgetting)

## Core capability metrics

### 1) One-shot acquisition
**Question:** After one exposure, does the system form a usable association?

Protocol:
- present stimulus S once
- apply a reinforcement signal to prefer action A
- test S again and record whether A is selected

Metric:
- one-shot correctness rate over N novel stimuli

### 2) Recall from partial cue
**Question:** With a weak/noisy stimulus, does the system still retrieve the right attractor?

Protocol:
- train associations with strong stimuli
- test with reduced stimulus strength (e.g. 30–50%)

Metric:
- partial-cue accuracy

### 3) Habit strength / consistency
**Question:** Under repeated conditions, does behavior become stable?

Protocol:
- repeat same stimulus for T steps
- measure how often the same action is produced

Metric:
- action stability = fraction of steps matching modal action

### 4) Adaptation / switching cost
**Question:** When context flips, how quickly does the system switch attractors?

Protocol:
- train/drive stimulus A for K steps, then stimulus B for K steps

Metrics:
- steps-to-switch (time to choose the new target action)

### 5) Forgetting half-life (relevance)
**Question:** Does unused knowledge decay to free capacity?

Protocol:
- train a set of associations
- run a long “idle” period with no relevant stimuli
- test associations periodically

Metrics:
- accuracy vs time
- connection count vs time (pruning rate)

## Energy / edge proxy metrics
We can’t measure power perfectly in software, but we can use proxies:
- `connections_updated_per_step` (dominant cost)
- `avg_amp_change_per_step` (activity)
- `connection_count` (memory footprint)

## Live operational signals (daemon/UI)
For long-running autonomy (minimal external influence), these are often more informative than raw counts:
- `pruned_last_step`: structural forgetting rate (should be >0 occasionally; too high may indicate instability)
- `births_last_step`: neurogenesis activity (should be 0 most of the time, non-zero when saturated)
- `saturated`: whether the substrate is in a high-load regime (auto-neurogenesis trigger condition)
- `causal_last_directed_edge_updates` / `causal_last_cooccur_edge_updates`: how much meaning/credit assignment is happening each tick

If these remain at zero while behavior changes, it usually indicates the loop isn’t committing observations or isn’t seeing symbols.

These proxies should decrease with pruning and stabilize with habits.

## Reproducibility
All metrics should be runnable deterministically with a fixed RNG seed.

Current implementation provides a simple assays runner: `cargo run -- assays`

## Per-game definitions

For what the UI/daemon metrics mean *per game* (and what “correct” represents in each task), see:
- [Metrics per game](../games/metrics-per-game.md)
