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
            blurb: "Safe, high-level controls for the substrate runtime.",
        },
        SectionSpec {
            section: ParamSection::Neurogenesis,
            title: "Neurogenesis",
            blurb: "Grow capacity by adding new units (increases memory/compute).",
        },
        SectionSpec {
            section: ParamSection::Dynamics,
            title: "Dynamics",
            blurb: "Continuous-time oscillator dynamics and global stabilization.",
        },
        SectionSpec {
            section: ParamSection::Noise,
            title: "Noise",
            blurb: "Inject variability for exploration; too much can destabilize.",
        },
        SectionSpec {
            section: ParamSection::Plasticity,
            title: "Plasticity",
            blurb: "Local learning rates: Hebbian updates, imprinting, forgetting.",
        },
        SectionSpec {
            section: ParamSection::Salience,
            title: "Salience",
            blurb: "How quickly usage/attention accumulates and decays.",
        },
        SectionSpec {
            section: ParamSection::PruningThresholds,
            title: "Pruning & thresholds",
            blurb: "Link formation thresholds and pruning maintenance controls.",
        },
    ]
}

pub fn param_specs() -> Vec<ParamSpec> {
    let d = default_brain_config();

    vec![
        // Dynamics
        ParamSpec {
            key: "dt",
            label: "dt",
            section: ParamSection::Dynamics,
            description: "Integration timestep for the continuous dynamics.",
            units: Some("seconds (sim)"),
            min: 0.001,
            max: 1.0,
            step: 0.001,
            default: d.dt,
            recommended: Some(RecommendedRange { min: 0.01, max: 0.25 }),
            when_to_change: "Increase if learning is too sluggish; decrease if oscillations look unstable.",
            risk: Risk::High,
            advanced: false,
        },
        ParamSpec {
            key: "base_freq",
            label: "base_freq",
            section: ParamSection::Dynamics,
            description: "Intrinsic oscillator frequency (baseline rhythm) for unit dynamics.",
            units: Some("Hz (sim)"),
            min: 0.0,
            max: 10.0,
            step: 0.05,
            default: d.base_freq,
            recommended: Some(RecommendedRange { min: 0.5, max: 2.5 }),
            when_to_change: "Adjust if activity is too static (raise) or too jittery (lower).",
            risk: Risk::Medium,
            advanced: false,
        },
        ParamSpec {
            key: "global_inhibition",
            label: "global_inhibition",
            section: ParamSection::Dynamics,
            description: "Global inhibition strength (suppresses overall activity to improve selectivity).",
            units: None,
            min: 0.0,
            max: 5.0,
            step: 0.01,
            default: d.global_inhibition,
            recommended: Some(RecommendedRange { min: 0.02, max: 0.20 }),
            when_to_change: "Increase if everything fires at once; decrease if the network goes quiet.",
            risk: Risk::High,
            advanced: false,
        },

        // Noise
        ParamSpec {
            key: "noise_amp",
            label: "noise_amp",
            section: ParamSection::Noise,
            description: "Amplitude noise injected into dynamics (exploration).",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.noise_amp,
            recommended: Some(RecommendedRange { min: 0.0, max: 0.05 }),
            when_to_change: "Increase if the system gets stuck in repetitive attractors; reduce if learning becomes erratic.",
            risk: Risk::High,
            advanced: false,
        },
        ParamSpec {
            key: "noise_phase",
            label: "noise_phase",
            section: ParamSection::Noise,
            description: "Phase jitter applied to oscillators (variability without amplitude spikes).",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.noise_phase,
            recommended: Some(RecommendedRange { min: 0.0, max: 0.03 }),
            when_to_change: "Increase if actions become overly deterministic; reduce if phase-locking never stabilizes.",
            risk: Risk::Medium,
            advanced: true,
        },

        // Plasticity
        ParamSpec {
            key: "hebb_rate",
            label: "hebb_rate",
            section: ParamSection::Plasticity,
            description: "Hebbian learning rate (strength of local coupling updates).",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.hebb_rate,
            recommended: Some(RecommendedRange { min: 0.02, max: 0.15 }),
            when_to_change: "Raise if the agent is not forming associations; lower if it overfits quickly or thrashes.",
            risk: Risk::High,
            advanced: false,
        },
        ParamSpec {
            key: "forget_rate",
            label: "forget_rate",
            section: ParamSection::Plasticity,
            description: "Forgetting rate applied to learned couplings.",
            units: Some("per step"),
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.forget_rate,
            recommended: Some(RecommendedRange { min: 0.0005, max: 0.01 }),
            when_to_change: "Increase if stale habits persist; decrease if learning won’t stick.",
            risk: Risk::Medium,
            advanced: false,
        },
        ParamSpec {
            key: "imprint_rate",
            label: "imprint_rate",
            section: ParamSection::Plasticity,
            description: "Imprinting rate for stabilizing high-salience concepts (engram formation).",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.imprint_rate,
            recommended: Some(RecommendedRange { min: 0.2, max: 0.8 }),
            when_to_change: "Increase if you want more one-shot consolidation; decrease if concepts ‘lock in’ too early.",
            risk: Risk::High,
            advanced: true,
        },
        ParamSpec {
            key: "causal_decay",
            label: "causal_decay",
            section: ParamSection::Plasticity,
            description: "Decay rate for causal memory edges (how fast causal links fade).",
            units: Some("per step"),
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.causal_decay,
            recommended: Some(RecommendedRange { min: 0.0005, max: 0.01 }),
            when_to_change: "Increase if causal graph stays cluttered with old links; decrease if causal structure never accumulates.",
            risk: Risk::Medium,
            advanced: true,
        },

        // Salience
        ParamSpec {
            key: "salience_decay",
            label: "salience_decay",
            section: ParamSection::Salience,
            description: "Decay rate for salience/attention (how quickly usage fades).",
            units: Some("per step"),
            min: 0.0,
            max: 0.1,
            step: 0.0001,
            default: d.salience_decay,
            recommended: Some(RecommendedRange { min: 0.0005, max: 0.01 }),
            when_to_change: "Increase if the system clings to old salient units; decrease if salience never persists.",
            risk: Risk::Medium,
            advanced: true,
        },
        ParamSpec {
            key: "salience_gain",
            label: "salience_gain",
            section: ParamSection::Salience,
            description: "Gain applied to salience updates (how quickly usage becomes salient).",
            units: None,
            min: 0.0,
            max: 5.0,
            step: 0.01,
            default: d.salience_gain,
            recommended: Some(RecommendedRange { min: 0.5, max: 2.0 }),
            when_to_change: "Increase if important symbols don’t stand out; decrease if everything becomes ‘important’.",
            risk: Risk::Low,
            advanced: false,
        },

        // Pruning & thresholds
        ParamSpec {
            key: "prune_below",
            label: "prune_below",
            section: ParamSection::PruningThresholds,
            description: "Pruning threshold: connections with |w| below this may be removed during maintenance.",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.001,
            default: d.prune_below,
            recommended: Some(RecommendedRange { min: 0.0003, max: 0.01 }),
            when_to_change: "Increase to keep the graph sparse; decrease if useful weak links keep disappearing.",
            risk: Risk::Medium,
            advanced: true,
        },
        ParamSpec {
            key: "coactive_threshold",
            label: "coactive_threshold",
            section: ParamSection::PruningThresholds,
            description: "Co-activity threshold for forming associations (how strongly units must co-activate to link).",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.coactive_threshold,
            recommended: Some(RecommendedRange { min: 0.45, max: 0.75 }),
            when_to_change: "Increase to reduce spurious links; decrease if it fails to bind even when patterns co-occur.",
            risk: Risk::High,
            advanced: true,
        },
        ParamSpec {
            key: "phase_lock_threshold",
            label: "phase_lock_threshold",
            section: ParamSection::PruningThresholds,
            description: "Threshold for considering oscillators phase-locked (binding stability gate).",
            units: None,
            min: 0.0,
            max: 1.0,
            step: 0.01,
            default: d.phase_lock_threshold,
            recommended: Some(RecommendedRange { min: 0.50, max: 0.80 }),
            when_to_change: "Increase if binding is too promiscuous; decrease if nothing ever phase-locks.",
            risk: Risk::High,
            advanced: true,
        },
    ]
}
