use serde::{Deserialize, Serialize};

/// A bounded, slow-loop advisor integration point.
///
/// This is intentionally *not* an action selector.
/// It produces small, clamped configuration nudges (e.g. exploration/meaning weighting)
/// and is designed to be driven by an external LLM later.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvisorConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Invoke cadence in completed trials.
    #[serde(default = "default_every_trials")]
    pub every_trials: u32,
    /// Advisor mode: "stub" (built-in heuristic) or "off".
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_every_trials() -> u32 {
    25
}

fn default_mode() -> String {
    "stub".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvisorContext {
    #[serde(default)]
    pub game: String,
    #[serde(default)]
    pub context_key: String,

    #[serde(default)]
    pub trials: u32,
    #[serde(default)]
    pub accuracy: f32,
    #[serde(default)]
    pub recent_rate: f32,

    #[serde(default)]
    pub last_reward: f32,

    #[serde(default)]
    pub exploration_eps: f32,
    #[serde(default)]
    pub meaning_alpha: f32,

    #[serde(default)]
    pub text_regime: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvisorAdvice {
    /// How long (in trials) the receiver should consider this advice “active”.
    #[serde(default = "default_ttl_trials")]
    pub ttl_trials: u32,

    /// Optional bounded deltas/targets. The daemon clamps on application.
    #[serde(default)]
    pub exploration_eps: Option<f32>,
    #[serde(default)]
    pub meaning_alpha: Option<f32>,

    #[serde(default)]
    pub rationale: String,
}

fn default_ttl_trials() -> u32 {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvisorReport {
    #[serde(default)]
    pub at_trials: u32,
    #[serde(default)]
    pub applied: bool,
    #[serde(default)]
    pub context: AdvisorContext,
    #[serde(default)]
    pub advice: AdvisorAdvice,
}

#[derive(Debug, Clone, Default)]
pub struct AdvisorRuntime {
    pub cfg: AdvisorConfig,
    last_invoked_at_trials: u32,
    last_context_key: String,
    last_text_regime: Option<u32>,
    pub last_report: Option<AdvisorReport>,
}

impl AdvisorRuntime {
    pub fn new_from_env() -> Self {
        let mut rt = Self::default();

        // BRAINE_ADVISOR=off|stub
        if let Ok(v) = std::env::var("BRAINE_ADVISOR") {
            let vv = v.trim().to_ascii_lowercase();
            if vv == "off" || vv == "0" || vv == "false" {
                rt.cfg.enabled = false;
                rt.cfg.mode = "off".to_string();
            } else {
                rt.cfg.enabled = true;
                rt.cfg.mode = vv;
            }
        }

        // BRAINE_ADVISOR_EVERY_TRIALS=25
        if let Ok(v) = std::env::var("BRAINE_ADVISOR_EVERY_TRIALS") {
            if let Ok(n) = v.trim().parse::<u32>() {
                rt.cfg.every_trials = n.max(1);
            }
        }

        rt
    }

    pub fn status(&self) -> AdvisorConfig {
        self.cfg.clone()
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.cfg.enabled = enabled;
        if !enabled {
            self.last_report = None;
        }
    }

    pub fn set_every_trials(&mut self, every_trials: u32) {
        self.cfg.every_trials = every_trials.max(1);
    }

    pub fn set_mode(&mut self, mode: String) {
        self.cfg.mode = mode;
    }

    pub fn should_invoke(&self, completed_trials: u32) -> bool {
        if !self.cfg.enabled {
            return false;
        }
        if self.cfg.mode.trim().eq_ignore_ascii_case("off") {
            return false;
        }
        let every = self.cfg.every_trials.max(1);
        if completed_trials < every {
            return false;
        }
        completed_trials.saturating_sub(self.last_invoked_at_trials) >= every
    }

    pub fn invoke_stub(&mut self, ctx: &AdvisorContext) -> AdvisorAdvice {
        // This is deliberately simple: a placeholder for an external LLM.
        // It reacts to performance collapse and to obvious distribution shift
        // signals (e.g., a text regime flip) by adjusting exploration.

        let mut rationale_parts: Vec<String> = Vec::new();
        let mut exploration_target: Option<f32> = None;
        let mut meaning_alpha_target: Option<f32> = None;

        let regime_changed = match (self.last_text_regime, ctx.text_regime) {
            (Some(a), Some(b)) => a != b,
            (None, Some(_)) => false,
            _ => false,
        };

        if regime_changed {
            rationale_parts
                .push("detected regime change; increasing exploration for adaptation".to_string());
            exploration_target = Some((ctx.exploration_eps + 0.10).min(0.45));
        } else if ctx.trials >= 20 && ctx.recent_rate < 0.55 {
            rationale_parts.push("recent performance low; increasing exploration".to_string());
            exploration_target = Some((ctx.exploration_eps + 0.05).min(0.40));
        } else if ctx.trials >= 20 && ctx.recent_rate > 0.85 {
            rationale_parts.push("recent performance high; annealing exploration".to_string());
            exploration_target = Some((ctx.exploration_eps * 0.85).max(0.02));
        }

        // Keep meaning_alpha stable by default; small nudge only when very stuck.
        if ctx.trials >= 40 && ctx.recent_rate < 0.45 {
            rationale_parts
                .push("very low performance; slightly increasing meaning weight".to_string());
            meaning_alpha_target = Some((ctx.meaning_alpha + 0.05).min(1.0));
        }

        let rationale = if rationale_parts.is_empty() {
            "no change".to_string()
        } else {
            rationale_parts.join("; ")
        };

        AdvisorAdvice {
            ttl_trials: 50,
            exploration_eps: exploration_target,
            meaning_alpha: meaning_alpha_target,
            rationale,
        }
    }

    pub fn invoke(&mut self, ctx: AdvisorContext, at_trials: u32, apply: bool) -> AdvisorReport {
        let advice = match self.cfg.mode.trim().to_ascii_lowercase().as_str() {
            "stub" => self.invoke_stub(&ctx),
            // Future: http / openai / local model endpoint.
            other => AdvisorAdvice {
                ttl_trials: 0,
                exploration_eps: None,
                meaning_alpha: None,
                rationale: format!("advisor mode '{other}' not implemented; no-op"),
            },
        };

        self.last_invoked_at_trials = at_trials;
        self.last_context_key = ctx.context_key.clone();
        self.last_text_regime = ctx.text_regime;

        let report = AdvisorReport {
            at_trials,
            applied: apply,
            context: ctx,
            advice,
        };
        self.last_report = Some(report.clone());
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_increases_exploration_on_low_recent_rate() {
        let mut rt = AdvisorRuntime::default();
        rt.cfg.enabled = true;
        rt.cfg.mode = "stub".to_string();

        let ctx = AdvisorContext {
            game: "spot".to_string(),
            context_key: "spot".to_string(),
            trials: 50,
            accuracy: 0.4,
            recent_rate: 0.4,
            last_reward: -0.05,
            exploration_eps: 0.1,
            meaning_alpha: 0.2,
            text_regime: None,
        };

        let a = rt.invoke_stub(&ctx);
        assert!(a.exploration_eps.is_some());
        assert!(a.exploration_eps.unwrap() > 0.1);
    }
}
