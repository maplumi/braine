# Experiments

## Log template
Copy/paste for each experiment:

- Date:
- Branch/commit:
- Hypothesis:
- Change:
- Protocol (stimuli/actions/reward schedule):
- Metrics collected:
- Result summary:
- Next step:

## Current backlog (minimal, high value)

1) Ensure sensor/action groups never overlap
- Reason: overlapping allocations create accidental coupling and muddy metrics.

2) Add deterministic seeding everywhere
- Reason: makes regressions and improvements measurable.

3) Add a small “task battery”
- 2–5 tasks only: association, partial recall, switching, forgetting.

4) Edge profile pass
- Move toward fixed-size buffers, avoid per-step allocations.

5) Persistence
- Save/restore couplings + unit states in a compact format.
