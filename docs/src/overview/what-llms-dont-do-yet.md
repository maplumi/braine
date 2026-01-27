# What this model does that LLMs don’t (yet)

This note is about **capabilities and constraints**, not hype.
It describes things this project targets that today’s LLMs typically do not provide *out of the box*.

---

## 1) True closed-loop learning from interaction

**Braine target:** learn directly from *doing* (stimulus → action → reward), updating every step.
- No dataset collection phase required.
- No “retrain the model” loop required to incorporate a new experience.

**Typical LLM reality (today):** LLMs can control tools, but learning from the new experience usually requires:
- external memory (logs / databases) + prompting, or
- an explicit training/fine-tuning step.

---

## 2) Continuous online adaptation with bounded compute

**Braine target:** continuous online adaptation under edge constraints:
- sparse local updates
- bounded per-step compute
- structural forgetting/pruning to manage capacity

**Typical LLM reality:** inference is expensive, and continual adaptation is usually not part of inference.
Even when online updates exist in research systems, they are not usually cheap or robust enough for “always-on edge learning.”

---

## 3) Persistent *stateful* behavior (memory as structure)

**Braine target:** the system’s behavior is a function of its persistent internal structure:
- couplings evolve
- causal/meaning links accumulate
- saving/restoring state is part of the normal runtime loop

**Typical LLM reality:** an LLM’s weights are usually fixed at inference time.
State is commonly handled by:
- context window (limited),
- external memory stores,
- agent frameworks.

Those are powerful, but they are *add-ons* rather than a single integrated dynamical system that naturally “becomes what it experienced.”

---

## 4) Native embodiment framing

**Braine target:** treat “inputs” as sensor events and “outputs” as actions.
- Sensors are named channels with strengths.
- Reward is a scalar neuromodulator.

**Typical LLM reality:** LLMs are primarily text-token predictors.
They can be wrapped into agents with sensors/actions, but the base objective is not inherently grounded in sensorimotor loops.

---

## 5) Temporal structure as first-class dynamics (not just tokens)

**Braine target:** time is intrinsic to the substrate (oscillatory phase + recurrence).
That supports experiments where timing and history matter without requiring an explicit sequence model trained on corpora.

**Typical LLM reality:** time/history is represented via tokens and attention over context.
This is strong for language and code, but not automatically the same thing as a continuously-evolving dynamical controller.

---

## Important caveats (what LLMs do better today)

LLMs are currently far stronger at:
- language understanding/generation, code synthesis, instruction following
- broad world knowledge (from training corpora)
- reasoning over text and structured representations

Braine is not trying to “beat LLMs” at those.
The aim is to build a minimal, edge-friendly **continual learning substrate** for interactive tasks, and then measure capability growth with a checklist/task battery.

---

## How we will keep this honest

We only claim a capability when it is **tested**.
- Use the checklist in [capabilities-checklist.md](../games/capabilities-checklist.md).
- Log experiments in [experiments.md](../games/experiments.md) with explicit determinism and encoder configs.
