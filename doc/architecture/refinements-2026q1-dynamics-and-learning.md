# Refinements (2026 Q1): Dynamics & Learning Rigor Improvements

This document proposes a set of targeted refinements motivated by external critique of the “Math Behind” writeup and by known stability/scaling risks.

Constraints (non-negotiable):
- Online learning only; no backprop; no global loss.
- Reuse the same recurrent dynamics for scale; do not spawn experts/child brains just because inputs are larger.
- Keep the daemon/service boundary intact (newline-delimited JSON over TCP).
- Preserve backward compatibility for saved brain images and protocol snapshots via defaults.

## Summary of the 7 items

1) Stabilize phase dynamics (add restoring nonlinearity)
2) Reduce amplitude overloading (introduce one slow activity trace)
3) Soften brittle thresholds (smooth coactivity/phase gates in eligibility)
4) Clarify/justify eligibility philosophy and scaling (doc + parameterization)
5) Refine neurogenesis triggers (use unmet demand/novelty signals)
6) Add temporal structure to meaning memory (lagged causal edges)
7) Presentation + rigor pass (equation cleanup, scaling notes, tests)

Each item below includes: motivation, proposed change, configuration knobs, implementation notes, and validation.

---

## 1) Stabilize phase dynamics

### Motivation
Current phase coupling is effective in toy settings but can become diffusion-like under noise if the coupling term lacks stable fixed points. This undermines phase-gated learning.

### Proposed change
Add an optional restoring nonlinearity in phase coupling. Keep current behavior as default for backward compatibility.

Add a config-controlled mapping from phase difference $\Delta\phi$ to a bounded coupling term:
- **Linear (current):** $f(\Delta\phi)=\Delta\phi$ (wrapped to $[-\pi,\pi]$)
- **Sinusoidal:** $f(\Delta\phi)=\sin(\Delta\phi)$
- **Tanh-linear:** $f(\Delta\phi)=\tanh(k\,\Delta\phi)$

### Config knobs
- `phase_coupling_mode: u8` (0=linear, 1=sin, 2=tanh)
- `phase_coupling_k: f32` (used only for tanh mode)

Defaults:
- `phase_coupling_mode = 1` (sin)
- `phase_coupling_k = 2.0`

### Implementation notes
- Location: `crates/core/src/core/substrate.rs` in the phase update path (all tiers: scalar/simd/parallel/gpu wrappers as applicable).
- Implement a shared helper used by all dynamics tiers to avoid “tier-dependent physics.”
- Keep existing `phase_alignment()` used for learning; this change is about dynamics coupling.

### Validation
- New unit test: phase coupling nonlinearity produces bounded influence for large $|\Delta\phi|$ (tanh/sin).
- Regression test: default config reproduces previous mode (linear).

---

## 2) Reduce amplitude overloading (add slow activity trace)

### Motivation
`amp` currently plays multiple roles (activation, thresholding, salience proxy, readout). This can create unintended feedback loops and makes tuning brittle.

### Proposed change
Introduce a slow low-pass activity trace $\bar a_i$ computed from `amp`.
Use $\bar a$ for:
- Eligibility coactivity magnitude (more stable than instantaneous amp)
- Salience updates (less threshold-edge noise sensitivity)
Optionally later: action readout smoothing.

### Config knobs
- `activity_trace_decay: f32` in $[0,1]$ where larger means faster decay.

Default:
- `activity_trace_decay = 0.05` (time constant ~20 steps)

### Implementation notes
- Add `activity_trace: Vec<f32>` to `Brain` state.
- Update trace during dynamics write-back (scalar/simd/parallel/gpu paths) since it’s derived state, not a learned weight.
- Modify eligibility and salience updates to use the trace as the primary activity signal.

### Validation
- Unit test: trace can drive eligibility even after instantaneous amp drops.
- Regression: existing learning tests continue passing (eligibility gating remains robust when tests directly set `amp`).

---

## 3) Soften brittle thresholds (eligibility gates)

### Motivation
Hard thresholds can create narrow learning regimes and make noise dominate near cutoffs.

### Proposed change
Keep the **deadband** for plasticity commit (it prevents drift), but soften:
- coactivity thresholding
- phase-lock thresholding

Coactivity: replace $(a-\theta)_+$ with a smooth function (softplus-like).
Phase-lock: replace stepwise `if align > kappa { align } else { -0.05 }` with a smooth interpolation:
- $\sigma = \mathrm{sigmoid}(s(\ell-\kappa))$
- $\mathrm{corr}= (1-\sigma)(-0.05) + \sigma\ell$

### Config knobs
- `coactive_softness: f32` (0 = hard, >0 = smoother)
- `phase_gate_softness: f32` (0 = hard, >0 = smoother)

Defaults:
- `coactive_softness = 0.05` (gentle softplus threshold)
- `phase_gate_softness = 0.05` (gentle sigmoid blend)

Compatibility note: when loading older brain images that lack these fields, the loader defaults to hard gates (0.0) to preserve prior behavior.

### Implementation notes
- Implement small numeric helpers (stable sigmoid/softplus) in `substrate.rs`.
- Apply only inside eligibility update; do not change dynamics gating.

### Validation
- Unit test: with softness>0, corr changes continuously around threshold.
- Behavioral check: softness=0 matches old logic.

---

## 4) Clarify learning philosophy + scaling arguments

### Motivation
Eligibility is intentionally “Hebbian-ish” but not classic $a_i a_j$. The bounds/clamps are engineering choices that should be justified.

### Proposed change
- Add a short rationale section explaining discretization choices (sparsity, interpretability, robustness).
- Add a “why these bounds” section: relate clamp ranges to keeping per-step updates bounded relative to damping/inhibition.

### Implementation notes
- Documentation-only, plus possibly renaming variables in docs for clarity (no code rename required).

### Validation
- No code changes required.

---

## 5) Refine neurogenesis criteria

### Motivation
Mean |w| is too blunt as a saturation proxy.

### Proposed change
Add additional triggers that reflect “unmet demand / novelty”:
- sustained high CSR fill and low pruning recovery
- persistent high eligibility magnitude + frequent commit but flat reward trend
- novelty signal from new symbols/pairs rate

Keep old trigger as a fallback until new signals prove stable.

### Config knobs
- `growth_policy_mode: u8` (0=legacy, 1=hybrid)
- thresholds for eligibility/commit frequency window sizes

Defaults:
- `growth_policy_mode = 0` (legacy)

### Implementation notes
- Use existing monitors (`LearningStats`) where possible; extend minimally.
- Keep growth within `Brain::should_grow` / `maybe_neurogenesis`.

### Validation
- Unit tests for “hybrid policy triggers under sustained saturation signals.”

---

## 6) Add temporal structure to meaning memory

### Motivation
Pure Markovian co-occurrence can bias toward salience over usefulness. A small temporal structure improves credit assignment.

### Proposed change
Maintain a short ring buffer of symbol sets for the last N ticks and add lagged causal edges with geometric decay:
- lag 1 behaves like today
- lag >1 adds weaker directed edges

### Config knobs
- `causal_lag_steps: u8` (1..=16 recommended)
- `causal_lag_decay: f32` in (0,1)

Defaults:
- `causal_lag_steps = 1` (current behavior)
- `causal_lag_decay = 0.7`

Impact guidance:
- Keeping `causal_lag_steps = 1` avoids behavior changes and keeps meaning memory sharply Markovian.
- Increasing to 4–8 can improve delayed credit assignment (better alignment to reward timing) but tends to introduce more spurious long-range correlations unless lagged contributions are strongly decayed and symbol fanout per tick is capped.

### Implementation notes
- Best location: in `Brain` observation pipeline (so we can weight edges by lag) and call into `CausalMemory` with weighted counts.
- Keep memory bounded: cap the number of symbols per tick used for lagged updates.

### Validation
- Unit test: lagged edges exist and diminish with lag.
- Ensure existing causal memory tests still pass.

---

## 7) Presentation/rigger pass + tests

### Motivation
Small mathematical ambiguities (commas implying multiplication) weaken credibility.

### Proposed change
- Normalize notation in math docs; ensure each clamp and parameter is defined once.
- Add 2–3 high-value tests for new knobs to prevent regressions.

### Validation
- `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test --workspace`.
