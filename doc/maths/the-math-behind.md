# The Math Behind Braine

This document extracts the **mathematical model** that the core substrate implements in code.
It is intentionally **equations-first**, with minimal narrative.

Scope:
- Substrate dynamics (amplitude/phase oscillators on a sparse graph)
- Local plasticity (Hebbian + phase-lock gating + reward modulation)
- Memory systems (causal/meaning memory; “engram” traces)
- Action readout and reward handling
- Persistence / BBI (brain image) format and its scaling
- CPU vs GPU execution implications

Non-scope:
- UI, games, protocol JSON (except where it defines the control loop semantics)

---

## 0. Hypothesis / Theory

### Closed-loop substrate hypothesis
A continuously running recurrent substrate can learn online if it has:

1) **Stateful dynamics** (recurrent, not feed-forward)

2) **Local plasticity** (no global loss / no backprop)

3) **A scalar neuromodulator** $m(t)$ that gates plasticity (reward/salience)

4) **A boundary symbol memory** that can bind *stimulus/action/reward* events into a compact meaning graph

The working assumption is that:

- “Habits” emerge as **attractors** of the dynamical system.
- “Goal-directed” behavior emerges when attractor preferences are biased by **meaning memory**.

---

## 1. State variables, units, and ranges

Let there be $N$ units.

For unit $i \in \{0,\dots,N-1\}$ at discrete step $t$:

- amplitude: $a_i(t) \in [-2,2]$
- phase: $\phi_i(t) \in [-\pi,\pi]$ (wrapped)
- bias: $b_i \in [-0.5,0.5]$ (slow, includes reinforcement)
- decay: $\lambda_i > 0$ (amplitude decay factor)
- salience: $s_i(t) \in [0,10]$ (visualization/importance proxy)

Derived (non-persisted) state:

- activity trace: $a_i^{tr}(t) \in [0,2]$ (slow, nonnegative; used for salience/learning gates)

Sparse directed couplings are stored as CSR edges:

- adjacency list for each source $i$: targets $j \in \mathcal{N}(i)$
- weights: $w_{ij}(t) \in [-1.5,1.5]$

External input is injected per step via a “pending input” vector:

- injected current: $u_i(t)$

Neuromodulator:

- $m(t) \in [-1,1]$ (clamped)

---

## 2. Time and phases (the control loop)

A **single control cycle** in the intended usage is:

1) apply stimulus symbols (inject $u_i$ for sensor units)
2) optionally set neuromodulator $m(t)$ for this cycle
3) step the substrate dynamics (updates $(a,\phi)$)
4) read out an action (habit or habit+meaning)
5) record action symbols (and optionally pair symbols)
6) optionally reinforce action bias
7) commit or discard the “observation” into causal memory

Key distinction:

- **Dynamics + weight learning** happen in `step()`.
- **Meaning/causal memory updates** happen in `commit_observation()`.

---

## 3. Substrate dynamics (wave-like update)

### 3.1 Helper functions
Angle wrapping:

$$
\text{wrap}(x) \in [-\pi,\pi] \quad\text{by adding/subtracting }2\pi\text{ until in range.}
$$

Angle difference:

$$
\Delta(\alpha,\beta) = \text{wrap}(\alpha-\beta)
$$

Phase alignment score:

$$
\mathrm{align}(\phi_i,\phi_j) = \mathrm{clip}_{[0,1]}\Big(1 - \frac{|\Delta(\phi_i,\phi_j)|}{\pi}\Big)
$$

### 3.2 Global inhibition (competition)

Define mean amplitude (mode-dependent):

- **Signed mean (mode 0, legacy)**: $\bar{a}(t) = \frac{1}{N} \sum_{k=0}^{N-1} a_k(t)$
- **Mean absolute (mode 1)**: $\bar{a}(t) = \frac{1}{N} \sum_{k=0}^{N-1} |a_k(t)|$
- **Rectified mean (mode 2)**: $\bar{a}(t) = \frac{1}{N} \sum_{k=0}^{N-1} \max(0, a_k(t))$

Global inhibition term:

$$
I(t) = g \cdot \bar{a}(t)
$$

where $g$ is `global_inhibition`.

**Notes on inhibition modes**:
- Signed mean (legacy) can cancel out when positive/negative activity balances, leading to runaway pockets.
- Mean absolute treats all activity as magnitude, robust but may suppress useful inhibitory structure.
- Rectified mean only counts excitatory activity (positive amplitudes), aligns with salience/homeostasis conventions.

### 3.3 Sparse neighbor influence
For unit $i$:

Amplitude influence:

$$
A_i(t) = \sum_{j \in \mathcal{N}(i)} w_{ij}(t)\, a_j(t)
$$

Phase influence (bounded coupling):

Let $\delta_{ij}(t) = \Delta(\phi_j(t),\phi_i(t))$.

The phase coupling uses a mode, gain, and overall strength (`phase_coupling_mode`, `phase_coupling_k`, `phase_coupling_gain`):

- legacy linear (mode 0): $f(\delta) = \delta$
- sinusoidal (mode 1): $f(\delta) = \sin(\delta)$
- saturating tanh (mode 2): $f(\delta) = \tanh(k\,\delta)$

Then:

$$
P_i(t) = \sum_{j \in \mathcal{N}(i)} w_{ij}(t)\,\gamma_\phi\, f\big(\delta_{ij}(t)\big)
$$

where $\gamma_\phi$ is `phase_coupling_gain` (default 1.0).

### 3.4 Discrete-time update
Parameters:

- timestep $\Delta t$ = `dt`
- base frequency $\omega_0$ = `base_freq`
- noise terms: $\xi^a_i(t) \sim U[-\eta_a,\eta_a]$, $\xi^\phi_i(t) \sim U[-\eta_\phi,\eta_\phi]`

Amplitude damping:

$$
D_i(t) = \lambda_i\, a_i(t)
$$

Amplitude increment:

$$
\Delta a_i(t) = \big(b_i + u_i(t) + A_i(t) - I(t) - D_i(t) + \xi^a_i(t)\big)\,\Delta t
$$

The implementation additionally includes a smooth cubic saturation term to provide
soft restoring forces that avoid relying solely on hard clipping. Denote the
saturation parameter $\beta\ge 0$ (`amp_saturation_beta`):

$$
S_i(t) = -\beta\,a_i(t)^3
$$

and the amplitude increment becomes

$$
\Delta a_i(t) = \big(b_i + u_i(t) + A_i(t) - I(t) - D_i(t) + S_i(t) + \xi^a_i(t)\big)\,\Delta t
$$

Phase increment:

$$
\Delta \phi_i(t) = \big(\omega_0 + P_i(t) + \xi^\phi_i(t)\big)\,\Delta t
$$

State update with clamping/wrapping:

$$
\begin{aligned}
 a_i(t+1) &= \mathrm{clip}_{[-2,2]}\big(a_i(t) + \Delta a_i(t)\big)\\
 \phi_i(t+1) &= \text{wrap}\big(\phi_i(t) + \Delta\phi_i(t)\big)
\end{aligned}
$$

Notes:
- The added cubic term $-\beta a^3$ produces a smooth attractor structure
  (interior fixed points) when $\beta>0$, reducing brittleness introduced by
  hard clipping. Typical recommended values: $\beta\in[0.02,0.3]$; the code
  default is $0.1$.
- Clipping to $[-2,2]$ is retained as a safety bound but is no longer the
  primary mechanism that creates stable attractors.

### 3.5 Salience (importance trace)
Salience uses the coactivity threshold $\theta$ = `coactive_threshold`.

First, define a slow activity trace (EMA) with decay $d$ = `activity_trace_decay`:

$$
 a_i^{tr}(t+1) = (1-d)\,a_i^{tr}(t) + d\,\max(0, a_i(t+1))
$$

If $d = 0$, the implementation effectively uses instantaneous $\max(0,a_i)$.

Let activation for salience be:

$$
\alpha_i(t+1) = \max(0, a_i^{tr}(t+1) - \theta)
$$

With decay $\rho$ = `salience_decay` and gain $\gamma$ = `salience_gain`:

$$
 s_i(t+1) = \mathrm{clip}_{[0,10]}\Big((1-\rho)\,s_i(t) + \gamma\,\alpha_i(t+1)\Big)
$$

---

## 4. Plasticity: local Hebbian learning with phase gating

Learning uses:

- coactivity threshold $\theta$ = `coactive_threshold`
- phase-lock threshold $\kappa$ = `phase_lock_threshold` (in $[0,1]$)
- base learning rate $\eta$ = `hebb_rate`

In the current implementation, learning is split into two phases:

1) **Eligibility trace update** (runs continuously; does not change weights)

2) **Neuromodulated commit** (changes weights only when neuromodulation is present)

### 4.1 Eligibility trace update

Let eligibility decay be $\rho_e$ = `eligibility_decay` and eligibility gain be $\gamma_e$ = `eligibility_gain`.

Define a soft-thresholded co-activity magnitude (using the activity trace when enabled):

$$
c_{ij}(t) = \sqrt{\max(0, a_i^{tr}(t) - \theta)\,\max(0, a_j^{tr}(t) - \theta)}
$$

Compute phase alignment $\ell_{ij}(t) = \mathrm{align}(\phi_i(t),\phi_j(t))$ and define a correlation term:

$$
\mathrm{corr}_{ij}(t) = \begin{cases}
\ell_{ij}(t) & \text{if } \ell_{ij}(t) > \kappa \\
0 & \text{otherwise}
\end{cases}
$$

Eligibility updates (with per-step decay and bounded accumulation) are:

$$
e_{ij}(t+1) = \mathrm{clip}_{[-2,2]}\Big((1-\rho_e)\,e_{ij}(t) + \gamma_e\,c_{ij}(t)\,\mathrm{corr}_{ij}(t)\Big)
$$

### 4.2 Neuromodulated commit (deadband-gated)

Let neuromodulator be $m(t)$ = `neuromod` and define a deadband $d$ = `learning_deadband`.

Weights update only when:

$$
|m(t)| > d
$$

When committing, the signed neuromodulator scales the eligible change (negative values drive LTD):

$$
\Delta w_{ij}(t) = \mathrm{clip}_{[-0.25,0.25]}\big(\eta\,m(t)\,e_{ij}(t)\big)
$$

Optionally, a per-step plasticity budget can cap total committed change (approximately an L1 cap across all $|\Delta w|$).

Weight update:

$$
 w_{ij}(t+1) = \mathrm{clip}_{[-1.5,1.5]}\big(w_{ij}(t) + \Delta w_{ij}(t)\big)
$$

Notes / gaps:
- This is “Hebbian-like” but does **not** multiply by $a_i a_j$ explicitly; it uses a hard threshold + phase alignment.
- The deadband prevents constant drift when $m(t) \approx 0$.
- Signed neuromodulation supports both potentiation and depression (LTP/LTD).

---

## 5. Forgetting, pruning, and “engram” traces

Let forget rate be $f$ = `forget_rate` and prune threshold be $\epsilon$ = `prune_below`.

### 5.1 Weight decay
Every valid edge decays multiplicatively:

$$
 w_{ij} \leftarrow (1-f)\,w_{ij}
$$

### 5.2 Engram edges (sensor \leftrightarrow concept)
Special-case edges between:

- sensor units (members of any sensor group)
- concept units (reserved but not in any sensor/action group)

For such edges, if $|w_{ij}| < \epsilon$ after decay, the weight is clamped to a minimal trace:

$$
 w_{ij} \leftarrow \mathrm{sign}(w_{ij})\,\epsilon
$$

This implements a “savings” effect: re-learning can be faster than learning from zero.

### 5.3 Pruning
For all other edges:

If $|w_{ij}| < \epsilon$ then the edge is removed (tombstoned):

$$
\text{target}_{ij} \leftarrow \text{INVALID},\quad w_{ij} \leftarrow 0
$$

A periodic compaction rebuilds CSR to remove tombstones.

---

## 6. One-shot imprinting (concept formation)

Imprinting happens during stimulus application (not during `step`).

Given a sensor group $G$ and stimulus strength $\sigma$:

Imprinting is skipped if:

- $\sigma < 0.4$, or
- existing outgoing strength from the group is already large:

$$
\sum_{i\in G} \sum_{j \in \mathcal{N}(i)} |w_{ij}| > 3.0
$$

Otherwise, choose a “quiet” unit $c$ (low $|a_c|$) not in $G$, reserve it as a concept, and add couplings:

$$
\begin{aligned}
 w_{ic} &\leftarrow \mathrm{clip}(w_{ic} + r, [-1.5,1.5]) \\
 w_{ci} &\leftarrow \mathrm{clip}(w_{ci} + 0.7r, [-1.5,1.5])
\end{aligned}
$$

where $r$ is `imprint_rate`.

Also bias is increased:

$$
 b_c \leftarrow b_c + 0.04
$$

---

## 7. Neurogenesis (capacity growth)

Neurogenesis adds new units when the substrate is “saturated”.

A simple saturation proxy is mean absolute weight magnitude:

$$
\overline{|w|} = \frac{1}{|E|}\sum_{(i\to j)\in E} |w_{ij}|
$$

If $\overline{|w|}$ exceeds a threshold, new units are appended.

Each new unit $n$ is initialized with:

- $a_n=0$
- $\phi_n \sim U[-\pi,\pi]$
- $b_n \approx 0.05$ (slightly excitable)
- random outgoing edges to existing units with small weights
- a small number of incoming edges from existing units with positive weights

This changes resource usage over time because $N$ grows.

---

## 8. Action readout and policy

### 8.1 Pure habit readout
For an action group $A_k$ with unit set $\mathcal{A}_k$:

$$
\mathrm{score}_k^\mathrm{habit} = \sum_{i\in \mathcal{A}_k} a_i
$$

The deterministic policy chooses $\arg\max_k \mathrm{score}_k$.

An $\epsilon$-greedy policy chooses random action with prob $\epsilon$.

### 8.2 Habit + meaning readout
The combined score mixes a normalized habit term plus a causal “meaning” term.

Habit normalization (negative amplitude treated as inactive):

$$
\mathrm{habitNorm}_k = \mathrm{clip}_{[0,1]}\Bigg(
\frac{\sum_{i\in \mathcal{A}_k} \max(0,a_i)}{2\,|\mathcal{A}_k|}
\Bigg)
$$

Meaning is computed from causal memory (Section 9):

- global meaning for action symbol $a_k$:

$$
M_k^\mathrm{global} = S(a_k, r_+) - S(a_k, r_-)
$$

- conditional meaning for pair symbol $p = \mathrm{pair}(\text{stimulus}, a_k)$:

$$
M_k^\mathrm{cond} = S(p, r_+) - S(p, r_-)
$$

Total meaning:

$$
M_k = 1.0\,M_k^\mathrm{cond} + 0.15\,M_k^\mathrm{global}
$$

Final score (with parameter $\alpha \in [0,20]$):

$$
\mathrm{score}_k = 0.5\,\mathrm{habitNorm}_k + \alpha\,M_k
$$

---

## 9. Meaning / causal memory

Causal memory stores exponentially decayed counts:

- base counts: $B_s(t)$ for each symbol $s$
- directed edge counts: $C_{a\to b}(t)$ for symbol transitions

Let decay be $\delta$ = `causal_decay`.

### 9.1 Observation update
When observing a set of symbols $\mathcal{S}(t)$:

Decay is applied to all existing counts:

$$
B_s \leftarrow (1-\delta)B_s,\quad C_{a\to b} \leftarrow (1-\delta)C_{a\to b}
$$

Then base counts are incremented:

$$
\forall s\in\mathcal{S}(t):\quad B_s \leftarrow B_s + 1
$$

Directed edges from previous symbol set $\mathcal{S}(t-1)$ to current are incremented:

$$
\forall a\in\mathcal{S}(t-1),\,\forall b\in\mathcal{S}(t):\quad C_{a\to b} \leftarrow C_{a\to b} + 1
$$

Additionally, same-tick co-occurrence adds symmetric “cheap meaning” edges:

$$
\forall a\neq b\in\mathcal{S}(t):\quad C_{a\to b} \leftarrow C_{a\to b} + 0.5
$$

### 9.2 Causal strength
A cheap strength measure is:

$$
S(a,b) = \mathrm{clip}_{[-1,1]}\big(P(b\mid a) - P(b)\big)
$$

Where:

$$
P(b\mid a) \approx \mathrm{clip}_{[0,1]}\Big(\frac{C_{a\to b}}{B_a}\Big),\quad
P(b) \approx \mathrm{clip}_{[0,1]}\Big(\frac{B_b}{\sum_x B_x}\Big)
$$

Interpretation:
- $S(a,b)>0$ means $a$ increases likelihood of $b$.
- $S(a,b)<0$ means $a$ predicts $b$ less than baseline.

---

## 10. Reward / neuromodulator and observation commitment

Neuromodulator is set as a scalar $m(t)$ and clamped:

$$
 m(t) \leftarrow \mathrm{clip}_{[-1,1]}(m(t))
$$

When committing an observation, reward is discretized into symbols:

$$
\begin{cases}
 m(t) > 0.2 \Rightarrow r_+ \in \mathcal{S}(t)\\
 m(t) < -0.2 \Rightarrow r_- \in \mathcal{S}(t)\\
 \text{otherwise: no reward symbol}
\end{cases}
$$

The current symbol set is deduplicated and then fed to causal memory.

If “holdout / eval” mode is desired, the system may **discard** the observation, skipping causal memory writes, while still running dynamics and action selection.

---

## 11. Persistence: BBI (brain image) format

There are two related persisted formats:

1) The **brain image** (substate state), magic `BRAINE01`.
2) The **daemon state file** wrapper, magic `BRSTATE{1|2}`, which stores the brain image plus extra runtime/expert state.

### 11.1 Brain image header

- magic: 8 bytes = `BRAINE01`
- version: little-endian u32
  - v1: raw chunk payloads
  - v2: each chunk payload is LZ4-compressed

### 11.2 Brain image chunks (conceptual model)

The persisted state is a tuple:

$$
\mathcal{I} = (\text{CFG},\text{PRNG},\text{STAT},\text{UNITS},\text{CSR},\text{MASKS},\text{SAL},\text{GROUPS},\text{SYMBOLS},\text{CAUSAL})
$$

Chunk inventory:

- `CFG0`: scalar config parameters
- `PRNG`: RNG state
- `STAT`: age in steps
- `UNIT`: unit scalars + CSR connections (compacted)
- `MASK`: reserved mask + learning-enabled mask
- `SALI`: salience array (back-compat)
- `GRPS`: sensor/action group definitions
- `SYMB`: symbol string table (ID ↔ name)
- `CAUS`: causal memory payload

### 11.3 UNIT payload (structural scaling)

UNIT payload contains:

- unit_count: u32
- for each unit: 4×f32 = (amp, phase, bias, decay)
- connection_count: u32 (valid edges only)
- offsets: (unit_count+1)×u32
- targets: connection_count×u32
- weights: connection_count×f32

This is $O(N+E)$ space.

### 11.4 Causal payload

Causal payload contains decayed counts:

- decay: f32
- base symbol map: $|B|$ entries of (sym u32, count f32)
- edge map: $|C|$ entries of (key u64 packed(from,to), count f32)
- prev_symbols list

This grows with symbol variety + observed transitions.
Decay + periodic pruning bounds it in steady-state.

---

## 12. CPU vs GPU execution: what changes (and what does not)

### 12.1 Execution tiers
The substrate supports tiers:

- Scalar (single-thread)
- SIMD (vectorize the dense arithmetic; sparse traversal remains scalar)
- Parallel (multi-thread via Rayon)
- GPU (offload the **dense** update step)

### 12.2 Complexity per tick
Let $E$ be the number of edges (connections).

- sparse influence accumulation is $\Theta(E)$
- dense per-unit update is $\Theta(N)$

In the GPU tier implemented here:

- sparse influence accumulation still happens on CPU: $\Theta(E)$
- GPU does the dense update: $\Theta(N)$ on GPU

Learning (Hebbian) remains on CPU because it traverses sparse edges irregularly.

### 12.3 Memory growth over time
On CPU, memory is dominated by:

$$
\text{RAM} \approx O(N) \;\text{units} + O(E) \;\text{CSR arrays} + O(|B|+|C|) \;\text{causal memory}
$$

With neurogenesis, $N$ and $E$ increase over time.

On GPU, additional buffers exist (conceptually):

- unit arrays (amp/phase/bias/decay)
- influences (amp/phase/noise)
- pending input

So total working memory increases by another $O(N)$, but the **persisted image format does not change**.

---

## 13. Gaps / open equations (what we are *trying* to model better)

These are explicit “known limitations” of the current math as implemented:

1) **No synaptic delays**: couplings are instantaneous; temporal structure is pushed into causal memory counts.

2) **Phase coupling form is not Kuramoto**: it uses $\Delta(\phi_j,\phi_i)$ directly rather than $\sin(\phi_j-\phi_i)$.

3) **Hebbian update is thresholded**: it does not use a continuous $a_i a_j$ term; it uses coactivity thresholding + phase alignment.

4) **Reward (neuromodulator) discretization**: causal memory sees reward only as a thresholded symbol ($r_+$ or $r_-$), while plasticity sees reward as a continuous gain.

5) **No explicit energy function**: stability/attractors are empirical; we do not yet have a clean Lyapunov-style guarantee.

6) **GPU acceleration is partial**: dynamics are accelerated, but learning remains CPU-bound; long-term goal could include sparse-learning kernels.

7) **Credit assignment still approximate**: meaning memory uses $P(b|a)-P(b)$ which is cheap but not a full value function / temporal difference method.

---

## 14. Minimal “do not miss” function mapping (math to code loop)

The conceptual loop corresponds to:

- stimulus injection: $u_i(t)$
- dynamics: $(a,\phi)\mapsto(a',\phi')$
- plasticity: $w\mapsto w'$
- forgetting: $w'\mapsto \tilde w$
- readout: $a'\mapsto \text{action}$
- meaning update: $(\mathcal{S}(t-1),\mathcal{S}(t))\mapsto (B,C)$
- persistence: serialize $(a,\phi,b,\lambda,w,\text{symbols},B,C)$

If any future refactor changes a term above, this doc should be updated to preserve an exact math↔code correspondence.

### 14.1 Control-loop function inventory (audit list)

This list is intentionally literal: it names the functions that implement each phase.

**Stimulus / boundary symbol formation**

- `Brain::define_sensor`, `Brain::define_action` (allocate/mark units; set action biases)
- `Brain::apply_stimulus` (injects $u_i(t)$, records stimulus symbol, may call imprinting)
- `Brain::apply_stimulus_inference` (injects $u_i(t)$ without symbol/imprinting writes)
- `Brain::note_symbol`, `Brain::intern` (symbol table maintenance)
- `Brain::note_action`, `Brain::note_action_index` (records action symbols)
- `Brain::note_compound_symbol`, `Brain::note_pair_index` (records pair::<stimulus>::<action> symbols)

**Dynamics (state update)**

- `Brain::step` (dispatches dynamics tier, clears one-tick input, runs learning, then forgetting/pruning)
- `Brain::step_inference` (dynamics only; skips learning + forgetting)
- `Brain::step_nonblocking` (wasm GPU two-phase variant; still results in the same update equations)
- `Brain::step_dynamics_scalar` / `step_dynamics_simd` / `step_dynamics_parallel` / `step_dynamics_gpu`

**Plasticity + structural forgetting**

- `Brain::update_eligibility_scalar` (Section 4.1)
- `Brain::apply_plasticity_scalar` (Section 4.2)
- `Brain::homeostasis_step` (slow stability support; optional)
- `Brain::forget_and_prune` (Section 5)
- CSR maintenance: `Brain::add_or_bump_csr`, `Brain::append_connection`, `Brain::compact_connections`

**Reward / neuromodulator + action bias reinforcement**

- `Brain::set_neuromodulator` (sets $m(t)$)
- `Brain::reinforce_action`, `Brain::reinforce_action_index` (bias update $b_i$ for action units)

**Action readout**

- `Brain::select_action` (habit-only)
- `Brain::select_action_with_meaning`, `Brain::select_action_with_meaning_index` (habit+meaning)
- `Brain::select_action_predictive` (habit+meaning+prediction)
- Helpers: `Brain::action_score_breakdown`, `pair_reward_edges`, `action_reward_edges` (introspection)

**Meaning / memory commit**

- `Brain::commit_observation` (discretizes reward to symbols and calls causal observe)
- `Brain::discard_observation` (skips meaning writes; used for holdout/eval)
- `CausalMemory::observe` / `CausalMemory::causal_strength` (Section 9)

**One-shot learning + capacity management**

- `Brain::imprint_if_novel` (Section 6)
- `Brain::should_grow`, `Brain::maybe_neurogenesis`, `Brain::grow_unit`, `Brain::grow_units`, `Brain::grow_for_group` (Section 7)
- Optional maintenance: `Brain::dream`, `dream_replay`, `idle_dream`, `idle_maintenance`, `attention_gate`, `reset_learning_gates`

**Persistence (BBI) and growth impact**

- Brain image: `Brain::save_image_to`, `save_image_to_with_version`, `load_image_from`, plus chunk writers/readers
- Causal image payload: `CausalMemory::write_image_payload`, `read_image_payload`
- Daemon wrapper state: `crates/brained/src/state_image.rs` (stores brain image chunk as BIMG inside BRSTATE1/2)
