# Braine Research Questions

This document catalogs the scientific questions this project aims to answer.

---

## Core Research Questions

### Q1: Can phase-coupled oscillators learn associations?

**Hypothesis**: When two groups of oscillators repeatedly co-activate (same phase, high amplitude), Hebbian learning will strengthen their connections, enabling one to activate the other.

**Test**: Present stimulus A, require response X, reward when correct. Does A eventually trigger X?

**Status**: ⬜ Not formally tested

**Prediction**: Yes, this is the basic claim. Should work with 10-50 training trials.

---

### Q2: Does neuromodulation improve learning?

**Hypothesis**: A global "reward signal" (neuromodulator) that multiplicatively gates Hebbian learning will cause faster acquisition of rewarded associations and extinction of unrewarded ones.

**Test**: Compare learning speed with neuromodulation on vs. fixed learning rate.

**Status**: ⬜ Not formally tested

**Prediction**: Neuromodulation should improve relevant learning by 2-5x.

---

### Q3: Can the brain adapt to regime changes?

**Hypothesis**: When the correct response changes (e.g., A→X becomes A→Y), the combination of:
- Forgetting (weight decay)
- Negative reinforcement (wrong action punished)
- Continued learning (new association forms)

...will allow the brain to adapt within a reasonable number of trials.

**Test**: Train to criterion, flip mapping, measure trials to re-criterion.

**Status**: ⬜ Not formally tested

**Prediction**: Should recover within 30-100 trials. Faster with burst mode.

---

### Q4: Does phase synchronization enable binding?

**Hypothesis**: When multiple stimuli should be processed together (e.g., "red" + "left"), phase-locking between their units creates a unified representation distinct from other combinations.

**Test**: Present compound stimuli (A+B), train A+B→X. Does A alone or B alone trigger X? (Should not, fully.)

**Status**: ⬜ Not formally tested

**Prediction**: Partial - some generalization expected, but combination should be stronger.

---

### Q5: Can offline replay (dreaming) consolidate memories?

**Hypothesis**: Running the network without external input, allowing activity to propagate based on learned weights, will strengthen important associations.

**Test**: Train partially, dream, test. Compare to train same amount without dreaming.

**Status**: ⬜ Not formally tested

**Prediction**: Dream should improve retention by 10-30%, especially for older memories.

---

### Q6: Does attention gating improve learning efficiency?

**Hypothesis**: Restricting Hebbian learning to only the most active units (top 10-20%) will:
- Reduce interference from irrelevant associations
- Focus resources on currently relevant stimuli
- Improve signal-to-noise in learning

**Test**: Compare learning with attention gating on vs. off, especially with distractors.

**Status**: ⬜ Not formally tested

**Prediction**: Attention should help when distractors present, may hurt simple tasks.

---

### Q7: Is there a "savings" effect?

**Hypothesis**: After learning A→X, unlearning it, and then relearning it, the second learning should be faster than the first (classical savings effect).

**Test**: Train, untrain (reversal), retrain. Compare time-to-criterion.

**Status**: ⬜ Not formally tested

**Prediction**: If causal memory works correctly, should see 20-50% savings.

---

### Q8: Can hierarchical structure emerge?

**Hypothesis**: Without explicit layers, will some units naturally become "hidden units" that mediate between sensors and actions, forming useful intermediate representations?

**Test**: Analyze connection weights after training. Are there units that receive from sensors and project to actions?

**Status**: ⬜ Not formally tested

**Prediction**: Unclear. May require larger networks or specific training regimes.

---

### Q9: What is the capacity limit?

**Hypothesis**: The number of distinct associations the brain can hold is limited by:
- Number of units
- Connection density
- Weight precision

**Test**: Train progressively more associations, measure when performance degrades.

**Status**: ⬜ Not formally tested

**Prediction**: For 160 units, maybe 20-50 distinct associations before interference.

---

### Q10: Can temporal patterns be learned?

**Hypothesis**: The phase dynamics of oscillators provide short-term memory, allowing the brain to learn sequences like "A then B then C".

**Test**: Sequence prediction task with 3-4 elements.

**Status**: ⬜ Not formally tested

**Prediction**: Should work for 2-3 steps, degrade rapidly beyond that.

---

## Comparative Questions

### Q11: How does this compare to Q-learning?

**Key differences**:
- Q-learning: Stores explicit value per state-action pair
- Braine: Values implicit in connection weights

**Test**: Same task, compare sample efficiency and final performance.

**Prediction**: Q-learning faster for small state spaces, braine may generalize better.

---

### Q12: What are the advantages over neural networks?

**Potential advantages**:
- No backprop needed (local learning only)
- Sparse computation (only active units compute)
- More biologically plausible
- Potentially better continual learning

**Test**: Train on task A, then task B. Compare forgetting of A.

**Prediction**: Braine should show less catastrophic forgetting.

---

## Open Questions (No Clear Prediction)

### Q13: Can phase encode information?

Can we use phase (not just amplitude) to represent information? E.g., early phase = certain, late phase = uncertain?

### Q14: What learning rates are optimal?

Current rates are hand-tuned. Is there a principled way to set them?

### Q15: Should connections be symmetric?

Currently A→B and B→A are separate. Would symmetric weights work better?

### Q16: Is inhibition necessary?

Current global inhibition is simple. Would local inhibitory populations help?

### Q17: Can this scale to 10,000 units?

Current tests use 160 units. What happens at larger scales?

---

## Experiment Log Template

```markdown
### Experiment: [Name]
**Date**: YYYY-MM-DD
**Question**: Q# from above
**Setup**: 
- Game: 
- Brain config:
- Duration:

**Procedure**:
1. ...
2. ...

**Results**:
- Metric 1: 
- Metric 2:

**Conclusion**:
- Supports/refutes hypothesis because...

**Next steps**:
- ...
```

---

## Priority Order for Testing

1. **Q1** (basic learning) - must work for anything else to matter
2. **Q2** (neuromodulation) - core mechanism
3. **Q3** (adaptation) - distinguishes from lookup table
4. **Q5** (dreaming) - novel mechanism
5. **Q6** (attention) - novel mechanism
6. **Q4** (binding) - theoretically interesting
7. **Q7** (savings) - evidence of structured memory
8. Rest as time allows

