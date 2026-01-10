# Problem sets → Braine trials (ingestion guide)

This project learns from **closed-loop interaction**, not batch datasets.
A “problem set” needs to be transformed into a stream of **trials** that look like:

1) Apply **stimuli** (sensors) that represent the current context/state.
2) Run substrate dynamics (`step`).
3) Choose an **action** from a fixed action set.
4) Emit a scalar **reward/neuromodulator**.
5) Commit or discard the observation (learning vs eval).

This matches the daemon pattern described in doc/architecture/interaction.md.

## Conceptual mapping

A problem set item usually has some structure like:

- Context: input features (text token, class label prompt, sensor readings, etc.)
- Allowed actions: candidates the system may choose (class labels, next tokens, moves)
- Scoring: reward function for the chosen action (dense shaping + sparse outcome)
- Episode boundaries: when to reset, and what “done” means

In Braine terms:

- **Context → stimuli**: convert features into named sensor symbols (possibly many).
- **Action space → `define_action`/allowed actions**: keep action names stable across runs.
- **Scoring → neuromodulator**: produce a scalar reward $r \in [-1,1]$.
- **Supervision signal → symbols**: record helpful compound symbols like `pair::<ctx>::<action>` so meaning/causal memory can learn state-conditional action values.

## Practical design rules

- Keep sensor namespaces disjoint from action namespaces (avoid accidental coupling).
- Prefer *factorized* sensors (multiple small symbols) over one giant combined symbol, unless you explicitly want the combined key.
- If you need generalization, avoid leaking the answer in the stimulus key.
- Use **eval/holdout mode** by suppressing learning writes (discard observation instead of commit).

## Encoding choices

### Discrete classification / multiple choice
- Stimuli: one-hot symbols for the prompt/context, plus optional regime symbols.
- Actions: one action per class/candidate.
- Reward: +1 for correct, -1 for incorrect; optionally add dense shaping if learning is slow.

### Sequence prediction (next token)
- Stimuli: current token (and optional regime).
- Actions: choose the next token.
- Reward: +1 if predicted token matches target, else -1.

### Control / continuous state
- Stimuli: population-coded axes (see SpotXY), or discretized bins (see Pong bins).
- Actions: small discrete set.
- Reward: dense shaping (distance-to-goal) + sparse outcome reward.

## Where this plugs into the code

- Daemon games: implement `apply_stimuli`, decide `allowed_actions`, compute reward, then commit/discard.
- Web games: same structure, but the runtime lives in the browser tab.

If you want an actual file format for problem sets (e.g. JSONL) and a converter CLI, call out the target use-case (classification? sequence? planning?) and we can add a minimal schema + loader.
