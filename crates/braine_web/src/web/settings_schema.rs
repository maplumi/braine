use braine::substrate::BrainConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamSection {
    BraineSettings,
    Neurogenesis,
    Dynamics,
    Noise,
    Plasticity,
    Salience,
    PruningThresholds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Risk {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecommendedRange {
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub section: ParamSection,
    pub description: &'static str,
    pub units: Option<&'static str>,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub default: f32,
    pub recommended: Option<RecommendedRange>,
    pub when_to_change: &'static str,
    pub risk: Risk,
    pub advanced: bool,
}

pub struct SectionSpec {
    pub section: ParamSection,
    pub title: &'static str,
    pub blurb: &'static str,
}

fn default_brain_config() -> BrainConfig {
    // Keep this aligned with `make_default_brain()` (web runtime):
    // - defaults from core
    // - causal_decay override
    BrainConfig {
        seed: Some(2026),
        causal_decay: 0.002,
        ..BrainConfig::default()
    }
}

pub fn sections_ordered() -> Vec<SectionSpec> {
    vec![
        SectionSpec {
            section: ParamSection::BraineSettings,
            title: "Braine Settings",
            blurb: "Safe, high-level controls for the substrate runtime and general operation.",
        },
        SectionSpec {
            section: ParamSection::Neurogenesis,
            title: "Neurogenesis",
            blurb: "Add new computational units to expand capacity. Increases memory usage and compute requirements.",
        },
        SectionSpec {
            section: ParamSection::Dynamics,
            title: "Dynamics",
            blurb: "Core oscillator dynamics, integration timestep, and global inhibition for activity regulation.",
        },
        SectionSpec {
            section: ParamSection::Noise,
            title: "Noise & Exploration",
            blurb: "Controlled randomness for behavioral exploration. Too much noise prevents stable learning.",
        },
        SectionSpec {
            section: ParamSection::Plasticity,
            title: "Plasticity & Learning",
            blurb: "Controls for local Hebbian learning, memory formation (imprinting), and forgetting rates.",
        },
        SectionSpec {
            section: ParamSection::Salience,
            title: "Salience & Attention",
            blurb: "How quickly importance/attention builds up during use and fades when units are inactive.",
        },
        SectionSpec {
            section: ParamSection::PruningThresholds,
            title: "Pruning & Connection Thresholds",
            blurb: "Minimum thresholds for forming associations and maintaining weak connections over time.",
        },
    ]
}

pub fn param_specs() -> Vec<ParamSpec> {
    let d = default_brain_config();

    vec![
        // Dynamics
        ParamSpec {
            key: "dt",
            label: "dt (Timestep)",
            section: ParamSection::Dynamics,
            description: "Integration timestep for continuous oscillator dynamics. Larger values speed up simulation but may cause instability.",
            units: Some("seconds (sim)"),
            min: 0.001,
            max: 1.0,
            step: 0.001,
            default: d.dt,
            recommended: Some(RecommendedRange { min: 0.01, max: 0.25 }),
            when_to_change: "Increase if learning progresses too slowly; decrease if you see erratic oscillations or numerical instability.",
            risk: Risk::High,
            advanced: false,
        },
        ParamSpec {
            key: "base_freq",
            label: "Base Frequency",
            section: ParamSection::Dynamics,
            description: "Intrinsic oscillation frequency for all units. Sets the baseline temporal rhythm of neural activity.",
            units: Some("Hz (sim)"),
            min: 0.0,
            max: 10.0,
            step: 0.05,
            default: d.base_freq,
            recommended: Some(RecommendedRange { min: 0.5, max: 2.5 }),
            when_to_change: "Increase if activity appears too static or sluggish; decrease if dynamics are too jittery or chaotic.",
            risk: Risk::Medium,
            advanced: false,
        },
        ParamSpec {
            key: "global_inhibition",
            label: "Global Inhibition",
            section: ParamSection::Dynamics,
            description: "Suppresses overall network activity to promote selectivity and winner-take-all dynamics. Higher values increase sparsity.",
            units: None,
            min: 0.0,
            max: 5.0,
            step: 0.01,
            default: d.global_inhibition,
            recommended: Some(RecommendedRange { min: 0.02, max: 0.20 }),
            when_to_change: "Increase if too many units activate simultaneously; decrease if network activity becomes too sparse or silent.",
            risk: Risk::High,
            advanced: false,
        },

        // Noise
        ParamSpec {
            key: "noise_amp",
            label: "Noise Amplitude",
            section: ParamSection::Noise,
            description: "Amplitude of random noise injected into dynamics for exploration and preventing local minima. Enables behavioral variability.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.noise_amp,
            recommended: Some(RecommendedRange { min: 0.0, max: 0.05 }),
            when_to_change: "Increase if the system gets stuck in repetitive behaviors; reduce if learning becomes too erratic or unstable.",
            risk: Risk::High,
            advanced: false,
        },
        ParamSpec {
            key: "noise_phase",
            label: "Phase Noise",
            section: ParamSection::Noise,
            description: "Random jitter applied to oscillator phases. Adds temporal variability without large amplitude swings.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.noise_phase,
            recommended: Some(RecommendedRange { min: 0.0, max: 0.03 }),
            when_to_change: "Increase if actions become overly deterministic; reduce if phase-locking patterns fail to stabilize.",
            risk: Risk::Medium,
            advanced: true,
        },

        // Plasticity
        ParamSpec {
            key: "hebb_rate",
            label: "Hebbian Learning Rate",
            section: ParamSection::Plasticity,
            description: "Rate of local synaptic weight changes based on co-activation. Core mechanism for association formation ('neurons that fire together wire together').",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.hebb_rate,
            recommended: Some(RecommendedRange { min: 0.02, max: 0.15 }),
            when_to_change: "Increase if the brain is not forming associations quickly enough; lower if it overfits to noise or thrashes between patterns.",
            risk: Risk::High,
            advanced: false,
        },
        ParamSpec {
            key: "forget_rate",
            label: "Forgetting Rate",
            section: ParamSection::Plasticity,
            description: "Passive decay applied to connection weights. Allows old, unused associations to fade over time, preventing memory saturation.",
            units: Some("per step"),
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.forget_rate,
            recommended: Some(RecommendedRange { min: 0.0005, max: 0.01 }),
            when_to_change: "Increase if stale habits persist and interfere with new learning; decrease if learned behaviors don't persist long enough.",
            risk: Risk::Medium,
            advanced: false,
        },
        ParamSpec {
            key: "imprint_rate",
            label: "Imprinting Rate",
            section: ParamSection::Plasticity,
            description: "One-shot consolidation rate for high-salience patterns. Enables rapid formation of stable memory traces (engrams) for important concepts.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.imprint_rate,
            recommended: Some(RecommendedRange { min: 0.2, max: 0.8 }),
            when_to_change: "Increase for faster one-shot learning of important events; decrease if concepts become 'locked in' prematurely without refinement.",
            risk: Risk::High,
            advanced: true,
        },
        ParamSpec {
            key: "causal_decay",
            label: "Causal Memory Decay",
            section: ParamSection::Plasticity,
            description: "Decay rate for edges in the causal memory graph. Controls how quickly action→outcome relationships fade without reinforcement.",
            units: Some("per step"),
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.causal_decay,
            recommended: Some(RecommendedRange { min: 0.0005, max: 0.01 }),
            when_to_change: "Increase if the causal graph retains too many outdated links; decrease if causal structure never accumulates or persists.",
            risk: Risk::Medium,
            advanced: true,
        },

        // Salience
        ParamSpec {
            key: "salience_decay",
            label: "Salience Decay",
            section: ParamSection::Salience,
            description: "Rate at which unit importance/attention fades when inactive. Higher values cause salience to drop faster, promoting turnover.",
            units: Some("per step"),
            min: 0.0,
            max: 0.1,
            step: 0.0001,
            default: d.salience_decay,
            recommended: Some(RecommendedRange { min: 0.0005, max: 0.01 }),
            when_to_change: "Increase if the system fixates on outdated important units; decrease if salience never builds up or persists.",
            risk: Risk::Medium,
            advanced: true,
        },
        ParamSpec {
            key: "salience_gain",
            label: "Salience Gain",
            section: ParamSection::Salience,
            description: "Amplification factor for salience accumulation during active use. Higher values make frequently-used units stand out more prominently.",
            units: None,
            min: 0.0,
            max: 5.0,
            step: 0.01,
            default: d.salience_gain,
            recommended: Some(RecommendedRange { min: 0.5, max: 2.0 }),
            when_to_change: "Increase if important patterns don't become prominent enough; decrease if everything appears equally 'important' without clear differentiation.",
            risk: Risk::Low,
            advanced: false,
        },

        // Pruning & thresholds
        ParamSpec {
            key: "prune_below",
            label: "Pruning Threshold",
            section: ParamSection::PruningThresholds,
            description: "Minimum connection weight to survive pruning. Connections weaker than this threshold may be removed during maintenance to keep the graph sparse.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.prune_below,
            recommended: Some(RecommendedRange { min: 0.0003, max: 0.01 }),
            when_to_change: "Increase to maintain a sparser, more efficient graph; decrease if useful weak connections keep getting pruned away.",
            risk: Risk::Medium,
            advanced: true,
        },
        ParamSpec {
            key: "coactive_threshold",
            label: "Co-activity Threshold",
            section: ParamSection::PruningThresholds,
            description: "Minimum simultaneous activation strength required for two units to form an associative link. Gates spurious connections from weak coincidences.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.coactive_threshold,
            recommended: Some(RecommendedRange { min: 0.45, max: 0.75 }),
            when_to_change: "Increase to reduce noise and prevent spurious associations; decrease if the brain fails to bind patterns that genuinely co-occur.",
            risk: Risk::High,
            advanced: true,
        },
        ParamSpec {
            key: "phase_lock_threshold",
            label: "Phase-Lock Threshold",
            section: ParamSection::PruningThresholds,
            description: "Threshold for considering two oscillators phase-synchronized. Acts as a stability gate for temporal binding and feature integration.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.phase_lock_threshold,
            recommended: Some(RecommendedRange { min: 0.50, max: 0.80 }),
            when_to_change: "Increase if binding appears too promiscuous or unstable; decrease if units never achieve stable phase-locking even when they should.",
            risk: Risk::High,
            advanced: true,
        },
    ]
}
