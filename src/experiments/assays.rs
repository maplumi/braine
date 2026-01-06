use braine::substrate::{ActionPolicy, Brain, BrainConfig, Stimulus};

#[derive(Debug, Clone)]
struct AssayReport {
    seed: u64,
    train_steps: usize,
    test_trials: usize,
    partial_strength: f32,

    assoc_accuracy_full: f32,
    assoc_accuracy_partial: f32,

    switch_steps_avg: f32,

    forget_idle_steps: usize,
    accuracy_after_forget: f32,

    // Policy comparison: habit-only vs meaning-guided.
    novel_habit_stability: f32,
    novel_meaning_stability: f32,
    novel_meaning_hint: Option<(String, f32)>,

    energy_proxy_updates: u64,
    final_connection_count: usize,
}

pub fn run() {
    let seed = 1u64;

    // Keep this small and edge-realistic.
    let mut brain = Brain::new(BrainConfig {
        unit_count: 96,
        connectivity_per_unit: 8,
        dt: 0.05,
        base_freq: 1.0,
        noise_amp: 0.015,
        noise_phase: 0.008,
        global_inhibition: 0.07,
        hebb_rate: 0.09,
        forget_rate: 0.0015,
        prune_below: 0.0008,
        coactive_threshold: 0.55,
        phase_lock_threshold: 0.6,
        imprint_rate: 0.6,
        seed: Some(seed),
        causal_decay: 0.01,
    });

    brain.define_action("approach", 6);
    brain.define_action("avoid", 6);
    brain.define_action("idle", 6);

    brain.define_sensor("vision_food", 6);
    brain.define_sensor("vision_threat", 6);

    let mut policy = ActionPolicy::Deterministic;

    // === Training ===
    let train_steps = 800;
    let mut energy_proxy_updates: u64 = 0;

    for t in 0..train_steps {
        let stim = if (t / 200) % 2 == 0 {
            Stimulus::new("vision_food", 1.0)
        } else {
            Stimulus::new("vision_threat", 1.0)
        };

        brain.apply_stimulus(stim);
        brain.set_neuromodulator(0.7);

        brain.step();

        let (action, _score) = brain.select_action(&mut policy);
        brain.note_action(&action);

        match (stim.name, action.as_str()) {
            ("vision_food", "approach") => brain.reinforce_action("approach", 0.8),
            ("vision_food", _) => brain.reinforce_action("approach", 0.3),
            ("vision_threat", "avoid") => brain.reinforce_action("avoid", 0.8),
            ("vision_threat", _) => brain.reinforce_action("avoid", 0.3),
            _ => {}
        }

        brain.commit_observation();

        let diag = brain.diagnostics();
        energy_proxy_updates += diag.connection_count as u64;
    }

    let test_trials = 120;
    let assoc_accuracy_full = association_accuracy(&mut brain, &mut policy, test_trials, 1.0);

    let partial_strength = 0.35;
    let assoc_accuracy_partial = association_accuracy(&mut brain, &mut policy, test_trials, partial_strength);

    let switch_steps_avg = switching_cost(&mut brain, &mut policy);

    let forget_idle_steps = 1500;
    for _ in 0..forget_idle_steps {
        brain.set_neuromodulator(0.0);
        brain.step();
        let diag = brain.diagnostics();
        energy_proxy_updates += diag.connection_count as u64;
    }
    let accuracy_after_forget = association_accuracy(&mut brain, &mut policy, test_trials, 1.0);

    let (novel_habit_stability, novel_meaning_stability, novel_meaning_hint) =
        meaning_guided_vs_habit(seed);

    let final_diag = brain.diagnostics();

    let report = AssayReport {
        seed,
        train_steps,
        test_trials,
        partial_strength,
        assoc_accuracy_full,
        assoc_accuracy_partial,
        switch_steps_avg,
        forget_idle_steps,
        accuracy_after_forget,
        novel_habit_stability,
        novel_meaning_stability,
        novel_meaning_hint,
        energy_proxy_updates,
        final_connection_count: final_diag.connection_count,
    };

    print_report(&report);
}

fn association_accuracy(
    brain: &mut Brain,
    policy: &mut ActionPolicy,
    trials: usize,
    strength: f32,
) -> f32 {
    let mut correct = 0usize;

    for t in 0..trials {
        let stim = if t % 2 == 0 {
            Stimulus::new("vision_food", strength)
        } else {
            Stimulus::new("vision_threat", strength)
        };

        brain.apply_stimulus(stim);
        brain.set_neuromodulator(0.0);
        brain.step();

        let (action, _score) = brain.select_action(policy);
        brain.note_action(&action);
        brain.commit_observation();

        let ok = match (stim.name, action.as_str()) {
            ("vision_food", "approach") => true,
            ("vision_threat", "avoid") => true,
            _ => false,
        };

        if ok {
            correct += 1;
        }
    }

    correct as f32 / trials as f32
}

fn switching_cost(brain: &mut Brain, policy: &mut ActionPolicy) -> f32 {
    let mut switch_steps = Vec::new();

    for _ in 0..6 {
        for _ in 0..80 {
            brain.apply_stimulus(Stimulus::new("vision_food", 1.0));
            brain.set_neuromodulator(0.0);
            brain.step();
            let (a, _) = brain.select_action(policy);
            brain.note_action(&a);
            brain.commit_observation();
        }

        let mut steps = 0usize;
        loop {
            steps += 1;
            brain.apply_stimulus(Stimulus::new("vision_threat", 1.0));
            brain.set_neuromodulator(0.0);
            brain.step();
            let (action, _score) = brain.select_action(policy);
            brain.note_action(&action);
            brain.commit_observation();
            if action == "avoid" || steps >= 200 {
                break;
            }
        }
        switch_steps.push(steps as f32);
    }

    let sum: f32 = switch_steps.iter().sum();
    sum / switch_steps.len() as f32
}

fn print_report(r: &AssayReport) {
    println!("braine assays");
    println!("seed={}", r.seed);
    println!("train_steps={}", r.train_steps);
    println!("test_trials={}", r.test_trials);
    println!("partial_strength={:.2}", r.partial_strength);
    println!("assoc_accuracy_full={:.3}", r.assoc_accuracy_full);
    println!("assoc_accuracy_partial={:.3}", r.assoc_accuracy_partial);
    println!("switch_steps_avg={:.1}", r.switch_steps_avg);
    println!("forget_idle_steps={}", r.forget_idle_steps);
    println!("accuracy_after_forget={:.3}", r.accuracy_after_forget);
    println!("novel_habit_stability={:.3}", r.novel_habit_stability);
    println!("novel_meaning_stability={:.3}", r.novel_meaning_stability);
    println!("novel_meaning_hint={:?}", r.novel_meaning_hint);
    println!("energy_proxy_updates={}", r.energy_proxy_updates);
    println!("final_connection_count={}", r.final_connection_count);
}

fn meaning_guided_vs_habit(seed: u64) -> (f32, f32, Option<(String, f32)>) {
    let habit = run_novel_reward_loop(seed.wrapping_add(10_000), false);
    let meaning = run_novel_reward_loop(seed.wrapping_add(10_000), true);
    (habit.0, meaning.0, meaning.1)
}

fn run_novel_reward_loop(seed: u64, meaning_guided: bool) -> (f32, Option<(String, f32)>) {
    let novel = "vision_novel";
    let target = "avoid";

    let mut brain = Brain::new(BrainConfig {
        unit_count: 96,
        connectivity_per_unit: 8,
        dt: 0.05,
        base_freq: 1.0,
        noise_amp: 0.015,
        noise_phase: 0.008,
        global_inhibition: 0.07,
        hebb_rate: 0.09,
        forget_rate: 0.0015,
        prune_below: 0.0008,
        coactive_threshold: 0.55,
        phase_lock_threshold: 0.6,
        imprint_rate: 0.6,
        seed: Some(seed),
        causal_decay: 0.01,
    });

    brain.define_action("approach", 6);
    brain.define_action("avoid", 6);
    brain.define_action("idle", 6);
    brain.define_sensor(novel, 6);

    let mut det = ActionPolicy::Deterministic;
    let steps = 220usize;
    let mut target_count = 0usize;

    for _ in 0..steps {
        brain.apply_stimulus(Stimulus::new(novel, 1.0));
        brain.step();

        let (action, _score) = if meaning_guided {
            brain.select_action_with_meaning(novel, 1.5)
        } else {
            brain.select_action(&mut det)
        };
        brain.note_action(&action);

        let reward = if action == target { 0.7 } else { -0.4 };
        brain.set_neuromodulator(reward);
        if action == target {
            target_count += 1;
        }
        if reward > 0.0 {
            brain.reinforce_action(target, 0.4);
        }

        brain.commit_observation();
    }

    let stability = target_count as f32 / steps as f32;
    let hint = brain.meaning_hint(novel);
    (stability, hint)
}
