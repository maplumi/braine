mod experiments;

use braine::substrate::{ActionPolicy, Brain, BrainConfig, Stimulus};
use braine::supervisor::{ChildConfigOverrides, ChildSpec, Supervisor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && (args[1] == "--help" || args[1] == "-h" || args[1] == "help") {
        print_help();
        return;
    }
    if args.len() >= 2 && args[1] == "assays" {
        experiments::assays::run();
        return;
    }
    if args.len() >= 2 && args[1] == "spawn-demo" {
        run_spawn_demo();
        return;
    }
    if args.len() >= 2 && args[1] == "autonomy-demo" {
        run_autonomy_demo();
        return;
    }
    if args.len() >= 2 && args[1] == "pong-demo" {
        run_pong_demo();
        return;
    }
    if args.len() >= 2 && args[1] == "spotxy-demo" {
        run_spotxy_demo();
        return;
    }

    if args.len() >= 2 {
        eprintln!("Unknown command: {}", args[1]);
        print_help();
        std::process::exit(2);
    }

    // Minimal demo:
    // - two recurring stimuli ("food" and "threat")
    // - the system will imprint concepts on first exposure
    // - repetition strengthens couplings and makes behavior consistent
    // - unused couplings decay and eventually prune

    let mut brain = Brain::new(BrainConfig {
        unit_count: 96,
        connectivity_per_unit: 8,
        dt: 0.05,
        base_freq: 1.0,
        noise_amp: 0.02,
        noise_phase: 0.01,
        global_inhibition: 0.06,
        hebb_rate: 0.08,
        forget_rate: 0.0015,
        prune_below: 0.0008,
        coactive_threshold: 0.55,
        phase_lock_threshold: 0.6,
        imprint_rate: 0.6,
        seed: None,
        causal_decay: 0.01,
    });

    // Declare actions: they are simply named readouts from dedicated unit groups.
    brain.define_action("approach", 6);
    brain.define_action("avoid", 6);
    brain.define_action("idle", 6);

    // Declare sensors: each stimulus excites a small subset of units.
    brain.define_sensor("vision_food", 6);
    brain.define_sensor("vision_threat", 6);

    // Run a simple "life" loop.
    // Determinism comes from attractor dominance; randomness comes from injected noise.
    // "Reward" is a scalar neuromodulator that scales Hebbian updates.
    let mut policy = ActionPolicy::EpsilonGreedy { epsilon: 0.1 };

    for t in 0..1200 {
        // Alternate blocks where one stimulus repeats, forming habits.
        let stim = if (t / 200) % 2 == 0 {
            Stimulus::new("vision_food", 1.0)
        } else {
            Stimulus::new("vision_threat", 1.0)
        };

        // Occasionally give partial/noisy perception (tests recall).
        let stim_strength = if t % 37 == 0 { 0.35 } else { stim.strength };
        let stim = Stimulus::new(stim.name, stim_strength);

        brain.apply_stimulus(stim);

        // Simple reinforcement signal for learning
        let reward_signal = 0.6;
        brain.set_neuromodulator(reward_signal);

        brain.step();

        // Choose an action from readouts.
        let (action, score) = brain.select_action(&mut policy);
        brain.note_action(&action);

        // Give action-specific feedback (shapes habits):
        // - if stimulus is food, reinforce approach group
        // - if stimulus is threat, reinforce avoid group
        match (stim.name, action.as_str()) {
            ("vision_food", "approach") => brain.reinforce_action("approach", 0.8),
            ("vision_food", "avoid") => brain.reinforce_action("avoid", -0.3),
            ("vision_threat", "avoid") => brain.reinforce_action("avoid", 0.8),
            ("vision_threat", "approach") => brain.reinforce_action("approach", -0.4),
            _ => {}
        }

        brain.commit_observation();

        if t % 50 == 0 {
            let diag = brain.diagnostics();
            let meaning = brain.meaning_hint(stim.name);
            println!(
                "t={t:4} stim={:<13} action={:<8} score={:+.3}  units={} conns={} pruned={} avg_amp={:.3}  meaning_hint={:?}",
                stim.name,
                action,
                score,
                diag.unit_count,
                diag.connection_count,
                diag.pruned_last_step,
                diag.avg_amp,
                meaning
            );
        }
    }
}

fn print_help() {
    println!("braine (brain-like substrate prototype)");
    println!("usage:");
    println!("  cargo run");
    println!("  cargo run -- assays");
    println!("  cargo run -- spawn-demo");
    println!("  cargo run -- autonomy-demo");
    println!("  cargo run -- pong-demo");
    println!("  cargo run -- spotxy-demo");
    println!("  cargo run -- --help");
}

fn run_pong_demo() {
    let mut brain = Brain::new(BrainConfig {
        unit_count: 128,
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
        seed: Some(123),
        causal_decay: 0.01,
    });

    experiments::env_pong::run_pong_demo(&mut brain, experiments::env_pong::PongConfig::default());
}

fn run_spotxy_demo() {
    // Minimal 2D position input (population-coded) with a 2-action decision rule.
    // This is an early spatial-awareness stepping stone.

    // Sizing note: this demo allocates many sensor groups (pos_x_*, pos_y_*),
    // so it uses a larger substrate than the tiny default demo.
    let mut brain = Brain::new(BrainConfig {
        unit_count: 256,
        connectivity_per_unit: 10,
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
        seed: Some(2026),
        causal_decay: 0.01,
    });

    experiments::env_spot_xy::run_spotxy_demo(
        &mut brain,
        experiments::env_spot_xy::SpotXYConfig::default(),
    );
}

fn run_autonomy_demo() {
    // Demonstrates autonomous behavior pattern:
    // - Parent encounters an unknown stimulus name
    // - Parent spawns a child sandbox to learn it
    // - Parent consolidates the child's useful structure
    // - Parent then responds to the new stimulus

    let mut parent = Brain::new(BrainConfig {
        unit_count: 128,
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
        seed: Some(42),
        causal_decay: 0.01,
    });

    parent.define_action("approach", 6);
    parent.define_action("avoid", 6);
    parent.define_action("idle", 6);

    parent.define_sensor("vision_food", 6);
    parent.define_sensor("vision_threat", 6);

    // Warm-up with known stimuli.
    let mut policy = ActionPolicy::EpsilonGreedy { epsilon: 0.05 };
    for t in 0..250 {
        let stim = if t % 2 == 0 {
            Stimulus::new("vision_food", 1.0)
        } else {
            Stimulus::new("vision_threat", 1.0)
        };
        parent.apply_stimulus(stim);
        parent.set_neuromodulator(0.2);
        parent.step();
        let (a, _) = parent.select_action(&mut policy);
        parent.note_action(&a);
        parent.commit_observation();
    }

    // Encounter a novel stimulus.
    let novel = "vision_new";
    println!("encounter novel stimulus: {novel}");

    // In a real system, the Frame/body would map a new sensor channel to a sparse group.
    // For the demo, we materialize the sensor group in the parent immediately.
    if !parent.has_sensor(novel) {
        parent.define_sensor(novel, 6);
    }

    // Spawn a child sandbox to learn what to do with the new signal.
    println!("spawning child sandbox...");

    let mut sup = Supervisor::new(parent);
    sup.spawn_child(
        ChildSpec {
            name: "auto_child".to_string(),
            budget_steps: 700,
            stimulus_name: novel.to_string(),
            target_action: "avoid".to_string(),
        },
        999,
        ChildConfigOverrides {
            noise_amp: 0.04,
            noise_phase: 0.02,
            hebb_rate: 0.15,
            forget_rate: 0.0012,
        },
    );

    for _ in 0..700 {
        sup.step_children();
    }

    let winner = sup.consolidate_best();
    println!("consolidation result: {:?}", winner);

    parent = sup.parent;

    // Post-consolidation verification loop:
    // Parent experiences the novel stimulus and gets reward feedback.
    let target = "avoid";
    let verify_steps = 200;
    let mut target_count = 0usize;

    for _ in 0..verify_steps {
        parent.apply_stimulus(Stimulus::new(novel, 1.0));
        parent.step();
        let (action, _score) = parent.select_action_with_meaning(novel, 1.5);
        parent.note_action(&action);

        let reward = if action == target { 0.7 } else { -0.4 };
        parent.set_neuromodulator(reward);
        if action == target {
            target_count += 1;
        }
        if reward > 0.0 {
            parent.reinforce_action(target, 0.4);
        }

        parent.commit_observation();
    }

    let stability = target_count as f32 / verify_steps as f32;
    let hint = parent.meaning_hint(novel);
    let pass = stability >= 0.6 && hint.is_some();

    println!("autonomy-demo verification:");
    println!("  target_action={target}");
    println!("  action_stability={:.3}", stability);
    println!("  meaning_hint={:?}", hint);
    println!("  result={}", if pass { "PASS" } else { "FAIL" });

    if !pass {
        std::process::exit(1);
    }
}

fn run_spawn_demo() {
    // Parent with stable identity.
    let mut parent = Brain::new(BrainConfig {
        unit_count: 128,
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
        seed: Some(7),
        causal_decay: 0.01,
    });

    parent.define_action("approach", 6);
    parent.define_action("avoid", 6);
    parent.define_action("idle", 6);
    parent.define_sensor("vision_food", 6);
    parent.define_sensor("vision_threat", 6);

    // Give the parent a little baseline life.
    let mut policy = ActionPolicy::EpsilonGreedy { epsilon: 0.05 };
    for t in 0..200 {
        let stim = if t % 2 == 0 {
            Stimulus::new("vision_food", 1.0)
        } else {
            Stimulus::new("vision_threat", 1.0)
        };
        parent.apply_stimulus(stim);
        parent.set_neuromodulator(0.2);
        parent.step();
        let (a, _) = parent.select_action(&mut policy);
        parent.note_action(&a);
        parent.commit_observation();
    }

    let mut sup = Supervisor::new(parent);

    // Spawn a few children with different exploration/plasticity.
    let spec = ChildSpec {
        name: "learn_vision_new".to_string(),
        budget_steps: 600,
        stimulus_name: "vision_new".to_string(),
        target_action: "avoid".to_string(),
    };

    sup.spawn_child(
        ChildSpec {
            name: "child_fast".to_string(),
            ..spec.clone()
        },
        100,
        ChildConfigOverrides {
            noise_amp: 0.03,
            noise_phase: 0.015,
            hebb_rate: 0.16,
            forget_rate: 0.0012,
        },
    );
    sup.spawn_child(
        ChildSpec {
            name: "child_slow".to_string(),
            ..spec.clone()
        },
        200,
        ChildConfigOverrides {
            noise_amp: 0.02,
            noise_phase: 0.010,
            hebb_rate: 0.11,
            forget_rate: 0.0010,
        },
    );
    sup.spawn_child(
        ChildSpec {
            name: "child_explore".to_string(),
            ..spec.clone()
        },
        300,
        ChildConfigOverrides {
            noise_amp: 0.045,
            noise_phase: 0.02,
            hebb_rate: 0.13,
            forget_rate: 0.0014,
        },
    );

    for _ in 0..spec.budget_steps {
        sup.step_children();
    }

    let scored = sup.score_children();
    println!("child scores (best first):");
    for (idx, score) in &scored {
        println!("  {} score={:+.3}", sup.children[*idx].name, score);
    }

    let before = sup.parent.diagnostics();
    let winner = sup.consolidate_best();
    let after = sup.parent.diagnostics();

    println!("consolidated winner={:?}", winner);
    println!(
        "parent conns: before={} after={} (pruned_last_step={})",
        before.connection_count, after.connection_count, after.pruned_last_step
    );
}
