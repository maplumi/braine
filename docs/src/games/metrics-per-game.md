# Per-game metrics (HUD + runtime)

This repo exposes a small set of **common learning metrics** across daemon-run games.
These show up in the desktop UI HUD, `runtime.json`, and the daemon protocol `Status` response.

## Common metrics (all daemon games)

These are computed by the shared `GameStats` helper and are intended to be **cheap** and **comparable** across games.

- **trials**: Number of *scored* trials so far (not frames). A “trial” is game-defined; most games score exactly one action per trial.
- **correct / incorrect**: Counts of trials where the chosen action matched the game’s “correct” label.
- **accuracy**: `correct / (correct + incorrect)`.
- **recent_rate**: Correct fraction over the most recent up-to-200 trials.
- **last_100_rate**: Correct fraction over the last 100 trials (or a smaller suffix when `< 100`).
- **learning_at_trial / learned_at_trial / mastered_at_trial**: Milestones based on `last_100_rate` once `trials >= 20`:
  - learning: `last_100_rate >= 0.70`
  - learned: `last_100_rate >= 0.85`
  - mastered: `last_100_rate >= 0.95`

Important nuance: **reward** and **correctness** are related but not always identical. Some games use dense shaping rewards; correctness is always defined by a game-specific label.

## Game-specific meaning of “correct”

### Spot
- **Correct**: chose `left` when the spot was left, otherwise `right`.
- **Reward**: `+1` correct, `-1` incorrect.

### Bandit
- **Correct**: chose the *currently best-probability* arm (`best_action`).
- **Reward**: stochastic; the best arm still produces occasional negative reward.
- Interpretation: Accuracy measures **policy preference** (choosing the best arm), not “reward rate”.

### Spot Reversal
- **Correct**: chose the correct arm *after applying the reversal mapping* (once reversal becomes active).
- **Reward**: `+1` correct, `-1` incorrect.
- Interpretation: Look for a dip near reversal and recovery afterward.

### SpotXY
- **Correct**: depends on mode:
  - `binary_x`: correct is `left` for `x < 0`, else `right`.
  - `grid`: correct is the label for the sampled target cell.
- **Reward**: `+1` correct, `-1` incorrect.
- Interpretation: In eval/holdout mode, learning writes are suppressed; accuracy should reflect retained skill.

### Maze
- **Trial boundary**: a “trial” is an **episode outcome**, not a single move.
- **Correct**: reached the goal before timing out.
- **Reward**: dense shaping per move (step cost + distance shaping + novelty), plus terminal reward/penalty.
- Interpretation: `trials` can grow slowly if episodes are long; use longer runs for stable estimates.

### Pong
- **Correct**: a **proxy label** based on whether the chosen action matches a heuristic “move paddle toward predicted intercept” bucket.
- **Reward**: primary learning signal is hit/miss events (plus small shaping/penalties).
- Interpretation: A flat “accuracy” does not necessarily mean no learning; it can also indicate the proxy is too strict/too loose for the current physics/cadence.

### Text (Next Token)
- **Correct**: chose the target next-token action for the current token under the current regime.
- **Reward**: typically `+1/-1` per trial plus regime/shift dynamics.
- Interpretation: If the vocabulary shifts, accuracy may dip and then recover.

### Replay
- **Correct**: chose the dataset trial’s `correct_action`.
- **Reward**: per-trial reward is driven by matching the dataset’s intended action.

## Where these are implemented

- Metric logic: [crates/shared/braine_games/src/stats.rs](../../crates/shared/braine_games/src/stats.rs)
- Per-game “correct” labels:
  - Spot: [crates/shared/braine_games/src/spot.rs](../../crates/shared/braine_games/src/spot.rs)
  - Bandit: [crates/shared/braine_games/src/bandit.rs](../../crates/shared/braine_games/src/bandit.rs)
  - Spot Reversal: [crates/shared/braine_games/src/spot_reversal.rs](../../crates/shared/braine_games/src/spot_reversal.rs)
  - SpotXY: [crates/shared/braine_games/src/spot_xy.rs](../../crates/shared/braine_games/src/spot_xy.rs)
  - Maze: [crates/shared/braine_games/src/maze.rs](../../crates/shared/braine_games/src/maze.rs)
  - Text: [crates/shared/braine_games/src/text_next_token.rs](../../crates/shared/braine_games/src/text_next_token.rs)
  - Replay: [crates/shared/braine_games/src/replay.rs](../../crates/shared/braine_games/src/replay.rs)
  - Pong (daemon wrapper): [crates/brained/src/game.rs](../../crates/brained/src/game.rs)
