# Language contract (text prediction without becoming an LLM)

This document is an **implementation guide** for adding "LLM-like" *functional behaviors* (text continuation, context sensitivity, abstraction/generalization) to **Braine** while explicitly *not* importing LLM mechanisms (transformers, token embeddings, backprop, global losses, internet-scale pretraining).

Braine is a **closed-loop learning substrate**: online dynamics + local plasticity + scalar neuromodulation. Text is treated as a temporally structured **stimulus–action stream**, not a batch optimization target.

## What we are matching (functional targets)

We only claim overlap on a narrow, testable set of behaviors:

- **Next-symbol inference**: produce a plausible next token given recent context.
- **Contextual association**: earlier symbols influence later outputs (short/medium range).
- **Generalization via structure**: similar patterns produce similar continuations.

We do **not** target:

- Internet-scale perplexity or broad fluency out of the box.
- A single static model that "knows everything".
- Token embeddings, attention, or gradient-trained weights.

## What we already have in the codebase

The current substrate already supports the key ingredients needed for text as an environment:

- **Stimuli**: `Stimulus { name, strength }` via `Brain::apply_stimulus`.
- **Actions**: named action groups + selection via:
  - `Brain::ranked_actions_with_meaning` / `select_action_with_meaning_index`
  - `Brain::select_action_predictive` (adds a learned prediction bonus using context-symbols with a prefix)
- **Credit assignment**:
  - `note_action`, `note_compound_symbol(["pair", ctx, action])`
  - `set_neuromodulator(reward)` + `reinforce_action(action, reward)` + `commit_observation()`
- **A working precedent**: the shared `SequenceGame` is already a next-token prediction task (tiny vocabulary) with regime shifts.

So the missing piece is not "how to do next token"; it’s a **language problem contract** + a scaling path that stays within Braine’s invariants.

## The language problem contract

A language task must declare, at the frame boundary:

### 1) Stimulus symbols

A text input stream is injected as a sequence of stimulus events.

Two layers are useful:

- **Dynamics coding (bounded sensors)**: what excites the recurrent substrate.
- **Identity coding (symbols)**: what is recorded into causal/meaning memory.

Why two layers? Because large vocabularies should not require unbounded growth in sensor groups.

Recommended coding options:

- **Byte/char sensors (simple, bounded)**
  - 256 byte sensors or ~128 ASCII-ish sensors.
  - Great for the first implementation.
- **Hashed token sensors (bounded, open vocab)**
  - Fixed sensor bank, e.g. `tok_hash_00..255`.
  - For each token, stimulate a small subset (k-of-n) based on hashes.
  - Record the true token string as a symbol event for causality.

### 2) Action symbols

Actions are "candidate next tokens".

Important constraint: actions are **named action groups**. A 50k-vocab "one action per token" is not the right first step.

Recommended staged approach:

- **Phase 1: byte-level actions**: 256 actions (one per byte) is manageable.
- **Phase 2: small subword lexicon**: hundreds to a few thousand actions.
- **Later**: hierarchical emission (multiple action steps per produced token) if needed.

### 3) Reward / neuromodulation

Reward must be scalar and bounded `[-1, +1]`. Typical sources:

- **Teacher-forced next-token correctness** (synthetic corpora / held-out sequences)
  - reward +1 when predicted token matches ground truth, -1 otherwise.
  - This is not "next-token loss"; it is *environmental feedback* per action.
- **User correction** (interactive mode)
  - reward +1 for accepted token, negative for correction/backspace.
- **Constraint satisfaction** (structured tasks)
  - e.g., bracket matching, JSON validity, simple grammar acceptance.

### 4) Temporal cadence

Timing replaces positional embeddings.

The contract must define:

- token injection cadence (one token per trial window)
- boundaries (BOS/EOS, sentence boundary events)
- evaluation mode (run dynamics + selection but call `discard_observation()`)

## Implementation path (phased, grounded in current code)

### Phase 0 — Define the scope and non-goals (design-only)

Deliverables:

- Decide **tokenization level** for v0: bytes or chars.
- Decide initial "LLM-like" behaviors to demonstrate:
  - next-token on tiny corpora
  - online adaptation to a distribution shift
- Decide explicit non-goals to keep in README/docs.

### Phase 1 — Minimal byte-level text predictor (shared game)

Goal: prove end-to-end text prediction as a game, with bounded action space.

Deliverables:

- Add a new shared game (mirrors `SequenceGame` structure):
  - `crates/shared/braine_games/src/text.rs` (behind `feature = "std"`)
  - `TextNextTokenGame` holds:
    - a byte corpus (or a small generated grammar)
    - a cursor, current token, correct next token
    - timing fields (`trial_frame`, `response_made`, `stats`)
  - `apply_stimuli(&mut Brain)` injects current token as sensor (byte-coded)
  - `score_action(action)` yields reward based on next-token match
- Add tests similar to `SequenceGame` verifying:
  - cursor advance
  - correctness mapping
  - regime/shift behavior (optional)

Notes:

- Start with **byte sensors** and **byte actions**. It’s the cleanest bounded baseline.

### Phase 2 — Wire into runtimes (web first, then daemon)

Goal: expose the task through an existing runner loop.

Deliverables:

- Web:
  - Add `WebGame::Text(TextNextTokenGame)` and `ensure_text_io()` similar to `ensure_sequence_io()` in `crates/braine_web/src/web.rs`.
  - UI: display current token, predicted token, reward.
- Daemon:
  - Add `ActiveGame::Text(TextNextTokenGame)` in `crates/brained/src/main.rs` and initialize sensors/actions at daemon init.
  - Extend protocol (newline-delimited JSON) to configure:
    - corpus selection
    - eval/learn mode
    - tokenization mode

### Phase 3 — Context beyond 1 token (use dynamics + symbols)

Goal: move from "bigram" to "short context" without attention.

Deliverables (choose one, then layer the other):

1) **Multi-token stimuli with decay**
- Inject the last $k$ tokens as stimuli with decaying strengths, e.g. $1.0, 0.7, 0.5, ...$.
- This directly uses recurrent dynamics for context integration.

2) **Context keys for meaning conditioning**
- Add a context symbol tag each step, like `txt_ctx::<t-2>::<t-1>`.
- Feed this into credit assignment using existing `pair::<ctx>::<action>` patterns.
- Keep the number of distinct context symbols bounded (e.g., discretize / hash).

Optional: leverage `Brain::select_action_predictive` with a dedicated prefix like `txt_ctx_`.

### Phase 4 — Learn from corrections (interactive language environment)

Goal: show continual learning in a way LLMs don’t naturally do.

Deliverables:

- CLI or UI mode where the user:
  - provides a prompt stream
  - sees predicted tokens
  - accepts/corrects
- Reward shaping:
  - accept → +reward
  - correction/backspace → -reward
  - optionally add "valid UTF-8" or "valid JSON" constraints as additional reward signals

### Phase 5 — “Reasoning-like” behaviors via structured text tasks

Goal: demonstrate compositional generalization without claiming LLM parity.

Prefer synthetic, interpretable tasks:

- bracket/paren completion
- JSON completion with schema constraints
- copy/reverse tasks
- tiny arithmetic with explicit symbols (e.g. `add_3_4 -> 7`)

All of these can be turned into trial streams with clear reward.

### Phase 6 — Scaling knobs and safeguards

Key issues to handle intentionally:

- **Action space size**: keep bounded; avoid 50k action groups.
- **Symbol explosion**: avoid creating unbounded `txt_ctx::<...>` symbols; use hashing or bounded discretization.
- **Evaluation mode**: always support `discard_observation()` holdout to measure generalization.
- **Experts/child brains**: only for novelty (new alphabet/language/domain), not longer prompts.

## Measurement (what we should log)

Avoid claiming perplexity; use substrate-native diagnostics:

- top-1 accuracy, top-k accuracy
- mean reciprocal rank (MRR) computed from `ranked_actions_with_meaning`
- adaptation speed after a controlled distribution shift
- causal graph salience: strength of `pair::<ctx>::<action>` edges
- bounded memory behavior: connection count, pruning rate

## Public-facing statement (safe wording)

“Braine can perform text prediction and language inference, but it does so as an embodied, dynamical system that learns temporal symbol relationships online, rather than as a pre-trained statistical model optimized via backpropagation.”
