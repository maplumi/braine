use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::causality::{CausalMemory, SymbolId};
use crate::prng::Prng;
use crate::storage;

pub type UnitId = usize;

#[derive(Debug, Clone)]
pub struct Connection {
    pub target: UnitId,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct Unit {
    // "Wave" state: amplitude + phase.
    pub amp: f32,
    pub phase: f32,

    pub bias: f32,
    pub decay: f32,

    // Sparse local couplings.
    pub connections: Vec<Connection>,
}

#[derive(Debug, Clone, Copy)]
pub struct BrainConfig {
    pub unit_count: usize,
    pub connectivity_per_unit: usize,

    pub dt: f32,
    pub base_freq: f32,

    pub noise_amp: f32,
    pub noise_phase: f32,

    // Competition: subtract proportional inhibition from all units.
    pub global_inhibition: f32,

    // Local learning/forgetting.
    pub hebb_rate: f32,
    pub forget_rate: f32,
    pub prune_below: f32,

    pub coactive_threshold: f32,

    // If two units are active and phase-aligned, strengthen more.
    // Range ~ [0, 1]: higher means "must be more aligned".
    pub phase_lock_threshold: f32,

    // One-shot concept formation strength (imprinting).
    pub imprint_rate: f32,

    // If set, makes behavior reproducible for evaluation.
    pub seed: Option<u64>,

    // Causality/meaning memory decay (0..1). Higher means faster forgetting.
    pub causal_decay: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Stimulus<'a> {
    pub name: &'a str,
    pub strength: f32,
}

impl<'a> Stimulus<'a> {
    pub fn new(name: &'a str, strength: f32) -> Self {
        Self { name, strength }
    }
}

#[derive(Debug, Clone)]
pub enum ActionPolicy {
    Deterministic,
    EpsilonGreedy { epsilon: f32 },
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub unit_count: usize,
    pub connection_count: usize,
    pub pruned_last_step: usize,
    pub avg_amp: f32,
}

#[derive(Debug, Clone)]
struct NamedGroup {
    name: String,
    units: Vec<UnitId>,
}

pub struct Brain {
    cfg: BrainConfig,
    units: Vec<Unit>,

    rng: Prng,

    reserved: Vec<bool>,

    // If false, unit's outgoing connections do not undergo learning updates.
    // Used to protect a parent identity subset in child brains.
    learning_enabled: Vec<bool>,

    // External "sensor" input is just injected current to some units.
    sensor_groups: Vec<NamedGroup>,
    action_groups: Vec<NamedGroup>,

    pending_input: Vec<f32>,

    // Neuromodulator scales learning ("reward", "salience").
    neuromod: f32,

    // Boundary symbol table for causality/meaning.
    symbols: HashMap<String, SymbolId>,
    symbols_rev: Vec<String>,
    active_symbols: Vec<SymbolId>,
    causal: CausalMemory,

    reward_pos_symbol: SymbolId,
    reward_neg_symbol: SymbolId,

    pruned_last_step: usize,

    age_steps: u64,

    telemetry: Telemetry,
}

#[derive(Debug, Clone, Default)]
struct Telemetry {
    enabled: bool,

    last_stimuli: Vec<SymbolId>,
    last_actions: Vec<SymbolId>,
    last_reinforced_actions: Vec<(SymbolId, f32)>,
    last_committed_symbols: Vec<SymbolId>,
}

impl Brain {
    pub fn new(cfg: BrainConfig) -> Self {
        let mut rng = Prng::new(cfg.seed.unwrap_or(1));

        let mut units = Vec::with_capacity(cfg.unit_count);
        for _ in 0..cfg.unit_count {
            units.push(Unit {
                amp: 0.0,
                phase: rng.gen_range_f32(-core::f32::consts::PI, core::f32::consts::PI),
                bias: 0.0,
                decay: 0.12,
                connections: Vec::new(),
            });
        }

        // Random sparse wiring (no matrices, no dense ops).
        for i in 0..cfg.unit_count {
            let mut conns = Vec::with_capacity(cfg.connectivity_per_unit);
            for _ in 0..cfg.connectivity_per_unit {
                let mut target = rng.gen_range_usize(0, cfg.unit_count);
                if target == i {
                    target = (target + 1) % cfg.unit_count;
                }
                let weight = rng.gen_range_f32(-0.15, 0.15);
                conns.push(Connection { target, weight });
            }
            units[i].connections = conns;
        }

        let pending_input = vec![0.0; cfg.unit_count];
        let reserved = vec![false; cfg.unit_count];
        let learning_enabled = vec![true; cfg.unit_count];

        let mut symbols: HashMap<String, SymbolId> = HashMap::new();
        let mut symbols_rev: Vec<String> = Vec::new();

        // Reserve reward symbols up front.
        let reward_pos_symbol = intern_symbol(&mut symbols, &mut symbols_rev, "reward_pos");
        let reward_neg_symbol = intern_symbol(&mut symbols, &mut symbols_rev, "reward_neg");

        let causal = CausalMemory::new(cfg.causal_decay);

        Self {
            cfg,
            units,
            sensor_groups: Vec::new(),
            action_groups: Vec::new(),
            pending_input,
            neuromod: 0.0,
            pruned_last_step: 0,
            rng,
            reserved,
            learning_enabled,

            symbols,
            symbols_rev,
            active_symbols: Vec::new(),
            causal,
            reward_pos_symbol,
            reward_neg_symbol,

            age_steps: 0,
            telemetry: Telemetry::default(),
        }
    }

    /// Enable/disable observer telemetry.
    /// When enabled, the brain records a small summary of what happened each loop.
    /// Observers read this data without mutating the functional state.
    pub fn set_observer_telemetry(&mut self, enabled: bool) {
        self.telemetry.enabled = enabled;
        if enabled {
            // Pre-allocate small buffers to avoid per-step allocations.
            if self.telemetry.last_stimuli.capacity() < 8 {
                self.telemetry.last_stimuli.reserve(8);
            }
            if self.telemetry.last_actions.capacity() < 8 {
                self.telemetry.last_actions.reserve(8);
            }
            if self.telemetry.last_reinforced_actions.capacity() < 8 {
                self.telemetry.last_reinforced_actions.reserve(8);
            }
            if self.telemetry.last_committed_symbols.capacity() < 16 {
                self.telemetry.last_committed_symbols.reserve(16);
            }
        }
    }

    pub fn age_steps(&self) -> u64 {
        self.age_steps
    }

    pub fn neuromodulator(&self) -> f32 {
        self.neuromod
    }

    pub fn causal_stats(&self) -> crate::causality::CausalStats {
        self.causal.stats()
    }

    pub fn symbol_name(&self, id: SymbolId) -> Option<&str> {
        self.symbols_rev.get(id as usize).map(|s| s.as_str())
    }

    pub fn last_stimuli_symbols(&self) -> &[SymbolId] {
        &self.telemetry.last_stimuli
    }

    pub fn last_action_symbols(&self) -> &[SymbolId] {
        &self.telemetry.last_actions
    }

    pub fn last_reinforced_action_symbols(&self) -> &[(SymbolId, f32)] {
        &self.telemetry.last_reinforced_actions
    }

    pub fn last_committed_symbols(&self) -> &[SymbolId] {
        &self.telemetry.last_committed_symbols
    }

    /// Serialize a versioned, chunked "brain image".
    ///
    /// This is std-only and intended to be capacity-aware when paired with
    /// `storage::CapacityWriter`.
    pub fn save_image_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(storage::MAGIC)?;
        storage::write_u32_le(w, storage::VERSION_V1)?;

        self.write_cfg_chunk(w)?;
        self.write_prng_chunk(w)?;
        self.write_stat_chunk(w)?;
        self.write_unit_chunk(w)?;
        self.write_mask_chunk(w)?;
        self.write_groups_chunk(w)?;
        self.write_symbols_chunk(w)?;
        self.write_causality_chunk(w)?;

        Ok(())
    }

    /// Load a versioned, chunked "brain image".
    ///
    /// Unknown chunks are skipped for forward-compatibility.
    pub fn load_image_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let magic = storage::read_exact::<8, _>(r)?;
        if &magic != storage::MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad brain image magic",
            ));
        }

        let version = storage::read_u32_le(r)?;
        if version != storage::VERSION_V1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported brain image version",
            ));
        }

        let mut cfg: Option<BrainConfig> = None;
        let mut rng_state: Option<u64> = None;
        let mut age_steps: Option<u64> = None;
        let mut units: Option<Vec<Unit>> = None;
        let mut reserved: Option<Vec<bool>> = None;
        let mut learning_enabled: Option<Vec<bool>> = None;
        let mut sensor_groups: Option<Vec<NamedGroup>> = None;
        let mut action_groups: Option<Vec<NamedGroup>> = None;
        let mut symbols_rev: Option<Vec<String>> = None;
        let mut causal: Option<CausalMemory> = None;

        loop {
            let (tag, len) = match storage::read_chunk_header(r) {
                Ok(v) => v,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            let mut take = r.take(len as u64);
            match &tag {
                b"CFG0" => {
                    cfg = Some(Self::read_cfg_payload(&mut take)?);
                }
                b"PRNG" => {
                    rng_state = Some(storage::read_u64_le(&mut take)?);
                }
                b"STAT" => {
                    age_steps = Some(storage::read_u64_le(&mut take)?);
                }
                b"UNIT" => {
                    units = Some(Self::read_unit_payload(&mut take)?);
                }
                b"MASK" => {
                    let (rsv, learn) = Self::read_mask_payload(&mut take)?;
                    reserved = Some(rsv);
                    learning_enabled = Some(learn);
                }
                b"GRPS" => {
                    let (sg, ag) = Self::read_groups_payload(&mut take)?;
                    sensor_groups = Some(sg);
                    action_groups = Some(ag);
                }
                b"SYMB" => {
                    symbols_rev = Some(Self::read_symbols_payload(&mut take)?);
                }
                b"CAUS" => {
                    causal = Some(CausalMemory::read_image_payload(&mut take)?);
                }
                _ => {
                    // Unknown chunk: skip.
                }
            }

            // Drain any remaining payload bytes for unknown or partially-read chunks.
            io::copy(&mut take, &mut io::sink())?;
        }

        let cfg = cfg.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing CFG0"))?;
        let unit_count = cfg.unit_count;
        let units =
            units.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing UNIT"))?;
        if units.len() != unit_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "UNIT count mismatch",
            ));
        }

        let reserved =
            reserved.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing MASK"))?;
        let learning_enabled = learning_enabled
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing MASK"))?;
        if reserved.len() != unit_count || learning_enabled.len() != unit_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MASK length mismatch",
            ));
        }

        let sensor_groups = sensor_groups
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing GRPS"))?;
        let action_groups = action_groups
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing GRPS"))?;

        let symbols_rev = symbols_rev
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing SYMB"))?;
        let (symbols, reward_pos_symbol, reward_neg_symbol) =
            Self::rebuild_symbol_tables(&symbols_rev)?;

        let causal =
            causal.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing CAUS"))?;
        let rng_state =
            rng_state.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing PRNG"))?;

        let age_steps = age_steps.unwrap_or(0);

        Ok(Self {
            cfg,
            units,
            rng: Prng::from_state(rng_state),
            reserved,
            learning_enabled,
            sensor_groups,
            action_groups,
            pending_input: vec![0.0; unit_count],
            neuromod: 0.0,
            symbols,
            symbols_rev,
            active_symbols: Vec::new(),
            causal,
            reward_pos_symbol,
            reward_neg_symbol,
            pruned_last_step: 0,
            age_steps,
            telemetry: Telemetry::default(),
        })
    }

    /// Exact serialized size in bytes for the current brain image.
    pub fn image_size_bytes(&self) -> io::Result<usize> {
        let mut cw = storage::CountingWriter::new();
        self.save_image_to(&mut cw)?;
        Ok(cw.written())
    }

    fn write_cfg_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload_len = Self::cfg_payload_len_bytes();
        w.write_all(b"CFG0")?;
        storage::write_u32_le(w, payload_len)?;
        self.write_cfg_payload(w)
    }

    fn cfg_payload_len_bytes() -> u32 {
        // Bytes based on exact write order below.
        4  // unit_count
            + 4  // connectivity_per_unit
            + 4  // dt
            + 4  // base_freq
            + 4  // noise_amp
            + 4  // noise_phase
            + 4  // global_inhibition
            + 4  // hebb_rate
            + 4  // forget_rate
            + 4  // prune_below
            + 4  // coactive_threshold
            + 4  // phase_lock_threshold
            + 4  // imprint_rate
            + 4  // seed_present
            + 8  // seed
            + 4 // causal_decay
    }

    fn write_cfg_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        storage::write_u32_le(w, self.cfg.unit_count as u32)?;
        storage::write_u32_le(w, self.cfg.connectivity_per_unit as u32)?;
        storage::write_f32_le(w, self.cfg.dt)?;
        storage::write_f32_le(w, self.cfg.base_freq)?;
        storage::write_f32_le(w, self.cfg.noise_amp)?;
        storage::write_f32_le(w, self.cfg.noise_phase)?;
        storage::write_f32_le(w, self.cfg.global_inhibition)?;
        storage::write_f32_le(w, self.cfg.hebb_rate)?;
        storage::write_f32_le(w, self.cfg.forget_rate)?;
        storage::write_f32_le(w, self.cfg.prune_below)?;
        storage::write_f32_le(w, self.cfg.coactive_threshold)?;
        storage::write_f32_le(w, self.cfg.phase_lock_threshold)?;
        storage::write_f32_le(w, self.cfg.imprint_rate)?;
        storage::write_u32_le(w, if self.cfg.seed.is_some() { 1 } else { 0 })?;
        storage::write_u64_le(w, self.cfg.seed.unwrap_or(0))?;
        storage::write_f32_le(w, self.cfg.causal_decay)?;
        Ok(())
    }

    fn read_cfg_payload<R: Read>(r: &mut R) -> io::Result<BrainConfig> {
        let unit_count = storage::read_u32_le(r)? as usize;
        let connectivity_per_unit = storage::read_u32_le(r)? as usize;

        let dt = storage::read_f32_le(r)?;
        let base_freq = storage::read_f32_le(r)?;
        let noise_amp = storage::read_f32_le(r)?;
        let noise_phase = storage::read_f32_le(r)?;
        let global_inhibition = storage::read_f32_le(r)?;
        let hebb_rate = storage::read_f32_le(r)?;
        let forget_rate = storage::read_f32_le(r)?;
        let prune_below = storage::read_f32_le(r)?;
        let coactive_threshold = storage::read_f32_le(r)?;
        let phase_lock_threshold = storage::read_f32_le(r)?;
        let imprint_rate = storage::read_f32_le(r)?;
        let seed_present = storage::read_u32_le(r)?;
        let seed = storage::read_u64_le(r)?;
        let causal_decay = storage::read_f32_le(r)?;

        Ok(BrainConfig {
            unit_count,
            connectivity_per_unit,
            dt,
            base_freq,
            noise_amp,
            noise_phase,
            global_inhibition,
            hebb_rate,
            forget_rate,
            prune_below,
            coactive_threshold,
            phase_lock_threshold,
            imprint_rate,
            seed: if seed_present != 0 { Some(seed) } else { None },
            causal_decay,
        })
    }

    fn write_prng_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(b"PRNG")?;
        storage::write_u32_le(w, 8)?;
        storage::write_u64_le(w, self.rng.state())
    }

    fn write_stat_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(b"STAT")?;
        storage::write_u32_le(w, 8)?;
        storage::write_u64_le(w, self.age_steps)
    }

    fn unit_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // unit_count
        for u in &self.units {
            len += 4 * 4; // amp,phase,bias,decay
            len += 4; // conn_count
            for _ in &u.connections {
                len += 4; // target
                len += 4; // weight
            }
        }
        Ok(u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "UNIT chunk too large"))?)
    }

    fn write_unit_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload_len = self.unit_payload_len_bytes()?;
        w.write_all(b"UNIT")?;
        storage::write_u32_le(w, payload_len)?;

        storage::write_u32_le(w, self.units.len() as u32)?;
        for u in &self.units {
            storage::write_f32_le(w, u.amp)?;
            storage::write_f32_le(w, u.phase)?;
            storage::write_f32_le(w, u.bias)?;
            storage::write_f32_le(w, u.decay)?;
            storage::write_u32_le(w, u.connections.len() as u32)?;
            for c in &u.connections {
                storage::write_u32_le(w, c.target as u32)?;
                storage::write_f32_le(w, c.weight)?;
            }
        }

        Ok(())
    }

    fn read_unit_payload<R: Read>(r: &mut R) -> io::Result<Vec<Unit>> {
        let unit_count = storage::read_u32_le(r)? as usize;
        let mut units: Vec<Unit> = Vec::with_capacity(unit_count);
        for _ in 0..unit_count {
            let amp = storage::read_f32_le(r)?;
            let phase = storage::read_f32_le(r)?;
            let bias = storage::read_f32_le(r)?;
            let decay = storage::read_f32_le(r)?;
            let conn_n = storage::read_u32_le(r)? as usize;
            let mut connections = Vec::with_capacity(conn_n);
            for _ in 0..conn_n {
                let target = storage::read_u32_le(r)? as usize;
                let weight = storage::read_f32_le(r)?;
                connections.push(Connection { target, weight });
            }
            units.push(Unit {
                amp,
                phase,
                bias,
                decay,
                connections,
            });
        }
        Ok(units)
    }

    fn mask_payload_len_bytes(&self) -> u32 {
        let n = self.units.len() as u32;
        let bytes_len = ((n as usize) + 7) / 8;
        // unit_count (u32) + reserved_len (u32) + reserved_bytes + learn_len (u32) + learn_bytes
        4 + 4 + bytes_len as u32 + 4 + bytes_len as u32
    }

    fn write_mask_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload_len = self.mask_payload_len_bytes();
        w.write_all(b"MASK")?;
        storage::write_u32_le(w, payload_len)?;

        let n = self.units.len();
        storage::write_u32_le(w, n as u32)?;

        let bytes_len = (n + 7) / 8;
        storage::write_u32_le(w, bytes_len as u32)?;
        Self::write_bool_bits(w, &self.reserved)?;

        storage::write_u32_le(w, bytes_len as u32)?;
        Self::write_bool_bits(w, &self.learning_enabled)?;

        Ok(())
    }

    fn write_bool_bits<W: Write>(w: &mut W, bits: &[bool]) -> io::Result<()> {
        let mut i = 0usize;
        while i < bits.len() {
            let mut byte = 0u8;
            for b in 0..8 {
                let idx = i + b;
                if idx >= bits.len() {
                    break;
                }
                if bits[idx] {
                    byte |= 1u8 << b;
                }
            }
            w.write_all(&[byte])?;
            i += 8;
        }
        Ok(())
    }

    fn read_mask_payload<R: Read>(r: &mut R) -> io::Result<(Vec<bool>, Vec<bool>)> {
        let n = storage::read_u32_le(r)? as usize;

        let reserved_len = storage::read_u32_le(r)? as usize;
        let reserved_bytes = {
            let mut buf = vec![0u8; reserved_len];
            r.read_exact(&mut buf)?;
            buf
        };

        let learning_len = storage::read_u32_le(r)? as usize;
        let learning_bytes = {
            let mut buf = vec![0u8; learning_len];
            r.read_exact(&mut buf)?;
            buf
        };

        if reserved_len != learning_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MASK byte length mismatch",
            ));
        }
        let expected_len = (n + 7) / 8;
        if reserved_len != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MASK byte length invalid",
            ));
        }

        Ok((
            Self::unpack_bool_bits(n, &reserved_bytes),
            Self::unpack_bool_bits(n, &learning_bytes),
        ))
    }

    fn unpack_bool_bits(n: usize, bytes: &[u8]) -> Vec<bool> {
        let mut out = vec![false; n];
        for i in 0..n {
            let byte = bytes[i / 8];
            let bit = (byte >> (i % 8)) & 1;
            out[i] = bit != 0;
        }
        out
    }

    fn groups_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // sensor group count
        for g in &self.sensor_groups {
            len += 4 + g.name.as_bytes().len() as u64;
            len += 4; // unit count
            len += 4 * g.units.len() as u64;
        }

        len += 4; // action group count
        for g in &self.action_groups {
            len += 4 + g.name.as_bytes().len() as u64;
            len += 4;
            len += 4 * g.units.len() as u64;
        }

        Ok(u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "GRPS chunk too large"))?)
    }

    fn write_groups_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload_len = self.groups_payload_len_bytes()?;
        w.write_all(b"GRPS")?;
        storage::write_u32_le(w, payload_len)?;

        storage::write_u32_le(w, self.sensor_groups.len() as u32)?;
        for g in &self.sensor_groups {
            storage::write_string(w, &g.name)?;
            storage::write_u32_le(w, g.units.len() as u32)?;
            for &u in &g.units {
                storage::write_u32_le(w, u as u32)?;
            }
        }

        storage::write_u32_le(w, self.action_groups.len() as u32)?;
        for g in &self.action_groups {
            storage::write_string(w, &g.name)?;
            storage::write_u32_le(w, g.units.len() as u32)?;
            for &u in &g.units {
                storage::write_u32_le(w, u as u32)?;
            }
        }

        Ok(())
    }

    fn read_groups_payload<R: Read>(r: &mut R) -> io::Result<(Vec<NamedGroup>, Vec<NamedGroup>)> {
        let sg_n = storage::read_u32_le(r)? as usize;
        let mut sensor_groups: Vec<NamedGroup> = Vec::with_capacity(sg_n);
        for _ in 0..sg_n {
            let name = storage::read_string(r)?;
            let n = storage::read_u32_le(r)? as usize;
            let mut units: Vec<UnitId> = Vec::with_capacity(n);
            for _ in 0..n {
                units.push(storage::read_u32_le(r)? as usize);
            }
            sensor_groups.push(NamedGroup { name, units });
        }

        let ag_n = storage::read_u32_le(r)? as usize;
        let mut action_groups: Vec<NamedGroup> = Vec::with_capacity(ag_n);
        for _ in 0..ag_n {
            let name = storage::read_string(r)?;
            let n = storage::read_u32_le(r)? as usize;
            let mut units: Vec<UnitId> = Vec::with_capacity(n);
            for _ in 0..n {
                units.push(storage::read_u32_le(r)? as usize);
            }
            action_groups.push(NamedGroup { name, units });
        }

        Ok((sensor_groups, action_groups))
    }

    fn symbols_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // count
        for s in &self.symbols_rev {
            len += 4 + s.as_bytes().len() as u64;
        }
        Ok(u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "SYMB chunk too large"))?)
    }

    fn write_symbols_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload_len = self.symbols_payload_len_bytes()?;
        w.write_all(b"SYMB")?;
        storage::write_u32_le(w, payload_len)?;

        storage::write_u32_le(w, self.symbols_rev.len() as u32)?;
        for s in &self.symbols_rev {
            storage::write_string(w, s)?;
        }
        Ok(())
    }

    fn read_symbols_payload<R: Read>(r: &mut R) -> io::Result<Vec<String>> {
        let n = storage::read_u32_le(r)? as usize;
        let mut out: Vec<String> = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(storage::read_string(r)?);
        }
        Ok(out)
    }

    fn rebuild_symbol_tables(
        symbols_rev: &[String],
    ) -> io::Result<(HashMap<String, SymbolId>, SymbolId, SymbolId)> {
        let mut symbols: HashMap<String, SymbolId> = HashMap::with_capacity(symbols_rev.len());
        let mut reward_pos: Option<SymbolId> = None;
        let mut reward_neg: Option<SymbolId> = None;

        for (i, name) in symbols_rev.iter().enumerate() {
            let id = i as SymbolId;
            if symbols.insert(name.clone(), id).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate symbol name",
                ));
            }
            if name == "reward_pos" {
                reward_pos = Some(id);
            }
            if name == "reward_neg" {
                reward_neg = Some(id);
            }
        }

        let reward_pos = reward_pos.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing reward_pos symbol")
        })?;
        let reward_neg = reward_neg.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing reward_neg symbol")
        })?;

        Ok((symbols, reward_pos, reward_neg))
    }

    fn write_causality_chunk<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let payload_len = self.causal.image_payload_len_bytes();
        w.write_all(b"CAUS")?;
        storage::write_u32_le(w, payload_len)?;
        self.causal.write_image_payload(w)
    }
    /// Ensure a sensor group exists; if missing, create it.
    pub fn ensure_sensor(&mut self, name: &str, width: usize) {
        if self.sensor_groups.iter().any(|g| g.name == name) {
            return;
        }
        self.define_sensor(name, width);
    }

    pub fn has_sensor(&self, name: &str) -> bool {
        self.sensor_groups.iter().any(|g| g.name == name)
    }

    /// Create a sandboxed child brain.
    ///
    /// Design intent:
    /// - child inherits structure (couplings + causal memory)
    /// - child can explore with different noise/plasticity
    /// - child cannot mutate a protected identity subset (action groups by default)
    pub fn spawn_child(
        &self,
        seed: u64,
        overrides: crate::supervisor::ChildConfigOverrides,
    ) -> Brain {
        let mut cfg = self.cfg;
        cfg.seed = Some(seed);
        cfg.noise_amp = overrides.noise_amp;
        cfg.noise_phase = overrides.noise_phase;
        cfg.hebb_rate = overrides.hebb_rate;
        cfg.forget_rate = overrides.forget_rate;

        let mut child = Brain::new(cfg);

        // Copy substrate state.
        child.units = self.units.clone();
        child.sensor_groups = self.sensor_groups.clone();
        child.action_groups = self.action_groups.clone();
        child.reserved = self.reserved.clone();

        // Copy symbol table + causal memory.
        child.symbols = self.symbols.clone();
        child.symbols_rev = self.symbols_rev.clone();
        child.reward_pos_symbol = self.reward_pos_symbol;
        child.reward_neg_symbol = self.reward_neg_symbol;
        child.causal = self.causal.clone();

        // Protect parent identity subset: action-group units.
        let mut mask = vec![true; child.units.len()];
        for g in &child.action_groups {
            for &id in &g.units {
                mask[id] = false;
            }
        }
        child.learning_enabled = mask;

        child
    }

    /// Consolidate structural/casual knowledge from a child back into self.
    /// Only merges strong, non-identity couplings.
    pub fn consolidate_from(
        &mut self,
        child: &Brain,
        policy: crate::supervisor::ConsolidationPolicy,
    ) {
        let thr = policy.weight_threshold;
        let rate = policy.merge_rate.clamp(0.0, 1.0);

        // Identity units are action group units.
        let mut protected = vec![false; self.units.len()];
        for g in &self.action_groups {
            for &id in &g.units {
                protected[id] = true;
            }
        }

        // Merge couplings.
        for i in 0..self.units.len() {
            if protected[i] {
                continue;
            }

            // For each child connection above threshold, pull parent weight toward child.
            for c_child in &child.units[i].connections {
                if c_child.weight.abs() < thr {
                    continue;
                }
                if c_child.target < protected.len() && protected[c_child.target] {
                    continue;
                }

                if let Some(c_parent) = self.units[i]
                    .connections
                    .iter_mut()
                    .find(|c| c.target == c_child.target)
                {
                    c_parent.weight = (1.0 - rate) * c_parent.weight + rate * c_child.weight;
                } else {
                    self.units[i].connections.push(c_child.clone());
                }
            }
        }

        // Merge causal memory: copy any strong edges from child.
        self.causal.merge_from(&child.causal, 0.25);
    }

    pub fn define_sensor(&mut self, name: &str, width: usize) {
        let units = self.allocate_units(width);
        self.sensor_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        self.intern(name);
    }

    pub fn define_action(&mut self, name: &str, width: usize) {
        let units = self.allocate_units(width);
        // Slight positive bias so actions can become stable attractors.
        for &id in &units {
            self.units[id].bias += 0.02;
        }
        self.action_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        self.intern(name);
    }

    pub fn apply_stimulus(&mut self, stimulus: Stimulus<'_>) {
        let group_units = self
            .sensor_groups
            .iter()
            .find(|g| g.name == stimulus.name)
            .map(|g| g.units.clone());

        if let Some(group_units) = group_units {
            for &id in &group_units {
                self.pending_input[id] += stimulus.strength;
            }

            self.note_symbol(stimulus.name);

            if self.telemetry.enabled {
                if let Some(id) = self.symbol_id(stimulus.name) {
                    self.telemetry.last_stimuli.push(id);
                }
            }

            // One-shot imprinting: when a stimulus is present, create a new "concept" unit
            // connected to currently active units (including the sensor group itself).
            // This is the simplest "instant learning" mechanism without training loops.
            self.imprint_if_novel(&group_units, stimulus.strength);
        }
    }

    /// Record the selected action as an event for causality/meaning.
    pub fn note_action(&mut self, action: &str) {
        self.note_symbol(action);

        if self.telemetry.enabled {
            if let Some(id) = self.symbol_id(action) {
                self.telemetry.last_actions.push(id);
            }
        }
    }

    /// Commit current perception/action/reward events into causal memory.
    /// Call this once per loop after:
    /// - apply_stimulus
    /// - step
    /// - select_action + note_action
    /// - (optional) reinforce_action
    pub fn commit_observation(&mut self) {
        // Map reward scalar to discrete events.
        if self.neuromod > 0.2 {
            self.active_symbols.push(self.reward_pos_symbol);
        } else if self.neuromod < -0.2 {
            self.active_symbols.push(self.reward_neg_symbol);
        }

        // Deduplicate cheaply (small vectors).
        self.active_symbols.sort_unstable();
        self.active_symbols.dedup();

        if self.telemetry.enabled {
            self.telemetry.last_committed_symbols.clear();
            self.telemetry
                .last_committed_symbols
                .extend_from_slice(&self.active_symbols);
        }

        self.causal.observe(&self.active_symbols);
        self.active_symbols.clear();
    }

    /// Very small "meaning" query: which action is most causally linked to positive reward
    /// under the last seen stimulus symbol.
    pub fn meaning_hint(&self, stimulus: &str) -> Option<(String, f32)> {
        let s = self.symbol_id(stimulus)?;

        let mut best: Option<(String, f32)> = None;
        for g in &self.action_groups {
            let a = self.symbol_id(&g.name)?;
            let score = self.causal.causal_strength(a, self.reward_pos_symbol)
                - self.causal.causal_strength(a, self.reward_neg_symbol);
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((g.name.clone(), score));
            }
        }

        // Also ensure stimulus is at least somewhat connected to the suggested action.
        best.and_then(|(act, sc)| {
            let a = self.symbol_id(&act)?;
            let link = self.causal.causal_strength(s, a);
            Some((act, sc * 0.7 + link * 0.3))
        })
    }

    /// Select an action using both:
    /// - current dynamical readout (habit/attractor)
    /// - learned meaning/causality (goal-directed)
    ///
    /// `alpha` weights meaning vs habit. `alpha=0` => pure habit.
    pub fn select_action_with_meaning(&mut self, stimulus: &str, alpha: f32) -> (String, f32) {
        // Allow meaning to dominate when needed (demo/goal-directed mode).
        let alpha = alpha.clamp(0.0, 20.0);
        let stimulus_id = self.symbol_id(stimulus);

        let mut best: Option<(String, f32)> = None;
        for g in &self.action_groups {
            // Habit readout: treat negative amplitude as "inactive" and normalize to ~[0,1].
            let habit = g
                .units
                .iter()
                .map(|&id| self.units[id].amp.max(0.0))
                .sum::<f32>();
            let habit_norm = if g.units.is_empty() {
                0.0
            } else {
                (habit / (g.units.len() as f32 * 2.0)).clamp(0.0, 1.0)
            };

            let meaning = if let Some(aid) = self.symbol_id(&g.name) {
                // Global action value (unconditional).
                let global = self.causal.causal_strength(aid, self.reward_pos_symbol)
                    - self.causal.causal_strength(aid, self.reward_neg_symbol);

                // State-conditional value using a composite symbol, if the environment emits it:
                //   pair::<stimulus>::<action>
                // This enables learning different actions for different stimuli without requiring
                // a dense model.
                let conditional = if stimulus_id.is_some() {
                    let pair_name = format!("pair::{stimulus}::{}", g.name);
                    if let Some(pid) = self.symbol_id(&pair_name) {
                        self.causal.causal_strength(pid, self.reward_pos_symbol)
                            - self.causal.causal_strength(pid, self.reward_neg_symbol)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // If conditional exists it should dominate; global is a weak prior.
                conditional * 1.0 + global * 0.15
            } else {
                0.0
            };

            // Keep habit as a weak tie-breaker; let meaning drive choice.
            let score = habit_norm * 0.5 + alpha * meaning;
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((g.name.clone(), score));
            }
        }

        best.unwrap_or_else(|| ("idle".to_string(), 0.0))
    }

    /// Return actions ranked by the same score used by `select_action_with_meaning`.
    ///
    /// Useful for visualization/debugging (e.g. showing top-N candidates in a HUD).
    pub fn ranked_actions_with_meaning(&self, stimulus: &str, alpha: f32) -> Vec<(String, f32)> {
        let alpha = alpha.clamp(0.0, 20.0);
        let stimulus_id = self.symbol_id(stimulus);

        let mut scored: Vec<(String, f32)> = Vec::with_capacity(self.action_groups.len());
        for g in &self.action_groups {
            let habit = g
                .units
                .iter()
                .map(|&id| self.units[id].amp.max(0.0))
                .sum::<f32>();
            let habit_norm = if g.units.is_empty() {
                0.0
            } else {
                (habit / (g.units.len() as f32 * 2.0)).clamp(0.0, 1.0)
            };

            let meaning = if let Some(aid) = self.symbol_id(&g.name) {
                let global = self.causal.causal_strength(aid, self.reward_pos_symbol)
                    - self.causal.causal_strength(aid, self.reward_neg_symbol);

                let conditional = if stimulus_id.is_some() {
                    let pair_name = format!("pair::{stimulus}::{}", g.name);
                    if let Some(pid) = self.symbol_id(&pair_name) {
                        self.causal.causal_strength(pid, self.reward_pos_symbol)
                            - self.causal.causal_strength(pid, self.reward_neg_symbol)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                conditional * 1.0 + global * 0.15
            } else {
                0.0
            };

            let score = habit_norm * 0.5 + alpha * meaning;
            scored.push((g.name.clone(), score));
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }

    pub fn top_actions_with_meaning(
        &self,
        stimulus: &str,
        alpha: f32,
        top_n: usize,
    ) -> Vec<(String, f32)> {
        let mut v = self.ranked_actions_with_meaning(stimulus, alpha);
        v.truncate(top_n);
        v
    }

    /// Explainability helper: strongest outgoing causal links from a named symbol.
    pub fn top_causal_links_from(&self, from: &str, top_n: usize) -> Vec<(String, f32)> {
        let Some(a) = self.symbol_id(from) else {
            return Vec::new();
        };

        self.causal
            .top_outgoing(a, top_n)
            .into_iter()
            .filter_map(|(bid, s)| self.symbol_name(bid).map(|name| (name.to_string(), s)))
            .collect()
    }

    pub fn set_neuromodulator(&mut self, value: f32) {
        // Clamp to a reasonable range.
        self.neuromod = value.clamp(-1.0, 1.0);
    }

    pub fn reinforce_action(&mut self, action: &str, delta_bias: f32) {
        if let Some(group) = self.action_groups.iter().find(|g| g.name == action) {
            if self.telemetry.enabled {
                if let Some(id) = self.symbol_id(action) {
                    self.telemetry
                        .last_reinforced_actions
                        .push((id, delta_bias));
                }
            }
            for &id in &group.units {
                self.units[id].bias = (self.units[id].bias + delta_bias * 0.01).clamp(-0.5, 0.5);
            }
        }
    }

    pub fn step(&mut self) {
        self.pruned_last_step = 0;

        self.age_steps = self.age_steps.wrapping_add(1);
        if self.telemetry.enabled {
            self.telemetry.last_stimuli.clear();
            self.telemetry.last_actions.clear();
            self.telemetry.last_reinforced_actions.clear();
        }

        // Compute global inhibition target as mean activity.
        let avg_amp = self.units.iter().map(|u| u.amp).sum::<f32>() / self.units.len() as f32;
        let inhibition = self.cfg.global_inhibition * avg_amp;

        let mut next_amp = vec![0.0; self.units.len()];
        let mut next_phase = vec![0.0; self.units.len()];

        for i in 0..self.units.len() {
            let u = &self.units[i];
            let mut influence_amp = 0.0;
            let mut influence_phase = 0.0;

            for c in &u.connections {
                let v = &self.units[c.target];
                // Wave-flavored coupling:
                // - amplitude is pulled by neighbor amplitude
                // - phase is gently pulled toward neighbor phase
                // These are local scalar ops; no matrices, no global objective.
                influence_amp += c.weight * v.amp;
                influence_phase += c.weight * angle_diff(v.phase, u.phase);
            }

            let noise_a = self
                .rng
                .gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp);
            let noise_p = self
                .rng
                .gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase);

            let input = self.pending_input[i];

            // Continuous-time-ish update.
            let damp = u.decay * u.amp;
            let d_amp =
                (u.bias + input + influence_amp - inhibition - damp + noise_a) * self.cfg.dt;
            let d_phase = (self.cfg.base_freq + influence_phase + noise_p) * self.cfg.dt;

            next_amp[i] = (u.amp + d_amp).clamp(-2.0, 2.0);
            next_phase[i] = wrap_angle(u.phase + d_phase);
        }

        for i in 0..self.units.len() {
            self.units[i].amp = next_amp[i];
            self.units[i].phase = next_phase[i];
        }

        // Clear one-tick inputs.
        for x in &mut self.pending_input {
            *x = 0.0;
        }

        self.learn_hebbian();
        self.forget_and_prune();
    }

    pub fn select_action(&mut self, policy: &mut ActionPolicy) -> (String, f32) {
        let mut scores: Vec<(String, f32)> = self
            .action_groups
            .iter()
            .map(|g| {
                (
                    g.name.clone(),
                    g.units.iter().map(|&id| self.units[id].amp).sum(),
                )
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        match policy {
            ActionPolicy::Deterministic => scores[0].clone(),
            ActionPolicy::EpsilonGreedy { epsilon } => {
                if self.rng.gen_range_f32(0.0, 1.0) < *epsilon {
                    let idx = self.rng.gen_range_usize(0, scores.len());
                    scores[idx].clone()
                } else {
                    scores[0].clone()
                }
            }
        }
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let connection_count = self
            .units
            .iter()
            .map(|u| u.connections.len())
            .sum::<usize>();
        let avg_amp = self.units.iter().map(|u| u.amp).sum::<f32>() / self.units.len() as f32;
        Diagnostics {
            unit_count: self.units.len(),
            connection_count,
            pruned_last_step: self.pruned_last_step,
            avg_amp,
        }
    }

    fn learn_hebbian(&mut self) {
        let thr = self.cfg.coactive_threshold;
        let lr = self.cfg.hebb_rate * (1.0 + self.neuromod.max(0.0));

        // Local rule: if i and j are co-active and phase-aligned, strengthen i->j.
        // Otherwise very slight anti-Hebb decay (encourages specialization).
        for i in 0..self.units.len() {
            if !self.learning_enabled[i] {
                continue;
            }
            let a_amp = self.units[i].amp;
            if a_amp <= thr {
                continue;
            }

            // Borrow-safely access unit i mutably and other units immutably.
            let (left, right) = self.units.split_at_mut(i);
            let (unit_i, right_rest) = right
                .split_first_mut()
                .expect("split_at_mut with valid index");

            let a_phase = unit_i.phase;
            for c in &mut unit_i.connections {
                let (b_amp, b_phase) = if c.target < i {
                    let b = &left[c.target];
                    (b.amp, b.phase)
                } else if c.target == i {
                    (unit_i.amp, unit_i.phase)
                } else {
                    let idx = c.target - i - 1;
                    let b = &right_rest[idx];
                    (b.amp, b.phase)
                };

                if b_amp > thr {
                    let align = phase_alignment(a_phase, b_phase);
                    if align > self.cfg.phase_lock_threshold {
                        c.weight += lr * align;
                    } else {
                        c.weight -= lr * 0.05;
                    }
                }

                c.weight = c.weight.clamp(-1.5, 1.5);
            }
        }
    }

    fn forget_and_prune(&mut self) {
        let decay = 1.0 - self.cfg.forget_rate;
        let prune_below = self.cfg.prune_below;

        for u in &mut self.units {
            for c in &mut u.connections {
                c.weight *= decay;
            }
            let before = u.connections.len();
            u.connections.retain(|c| c.weight.abs() >= prune_below);
            self.pruned_last_step += before - u.connections.len();
        }
    }

    fn allocate_units(&mut self, n: usize) -> Vec<UnitId> {
        // Choose from currently unreserved units only.
        let mut idxs: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.reserved[*i])
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();
        idxs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let chosen: Vec<UnitId> = idxs.into_iter().take(n).map(|(i, _)| i).collect();
        for &id in &chosen {
            self.reserved[id] = true;
        }
        chosen
    }

    fn imprint_if_novel(&mut self, group_units: &[UnitId], strength: f32) {
        // If the stimulus is weak, don't imprint.
        if strength < 0.4 {
            return;
        }

        // Detect novelty by checking whether sensor units already have strong outgoing couplings.
        let mut existing_strength = 0.0;
        for &id in group_units {
            existing_strength += self.units[id]
                .connections
                .iter()
                .map(|c| c.weight.abs())
                .sum::<f32>();
        }

        if existing_strength > 3.0 {
            return;
        }

        // Choose a "concept" unit: the quietest one not in the sensor group.
        let mut candidates: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !group_units.contains(i))
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let Some((concept_id, _)) = candidates.into_iter().next() else {
            return;
        };

        // Connect sensor units to the concept (and back) so it can be recalled.
        for &sid in group_units {
            add_or_bump_connection(
                &mut self.units[sid].connections,
                concept_id,
                self.cfg.imprint_rate,
            );
            add_or_bump_connection(
                &mut self.units[concept_id].connections,
                sid,
                self.cfg.imprint_rate * 0.7,
            );
        }

        // Make the concept slightly excitable.
        self.units[concept_id].bias += 0.04;
    }

    fn intern(&mut self, name: &str) -> SymbolId {
        intern_symbol(&mut self.symbols, &mut self.symbols_rev, name)
    }

    fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbols.get(name).copied()
    }

    fn note_symbol(&mut self, name: &str) {
        let id = self.intern(name);
        self.active_symbols.push(id);
    }
}

fn intern_symbol(
    map: &mut HashMap<String, SymbolId>,
    rev: &mut Vec<String>,
    name: &str,
) -> SymbolId {
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = rev.len() as SymbolId;
    rev.push(name.to_string());
    map.insert(name.to_string(), id);
    id
}

fn add_or_bump_connection(conns: &mut Vec<Connection>, target: UnitId, bump: f32) {
    if let Some(c) = conns.iter_mut().find(|c| c.target == target) {
        c.weight = (c.weight + bump).clamp(-1.5, 1.5);
    } else {
        conns.push(Connection {
            target,
            weight: bump.clamp(-1.5, 1.5),
        });
    }
}

fn wrap_angle(mut x: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    while x > core::f32::consts::PI {
        x -= two_pi;
    }
    while x < -core::f32::consts::PI {
        x += two_pi;
    }
    x
}

fn angle_diff(a: f32, b: f32) -> f32 {
    wrap_angle(a - b)
}

fn phase_alignment(a: f32, b: f32) -> f32 {
    // 1.0 when aligned, ~0.0 when opposite.
    let d = angle_diff(a, b).abs();
    let x = 1.0 - (d / core::f32::consts::PI);
    x.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_image_roundtrip_basic() {
        let cfg = BrainConfig {
            unit_count: 32,
            connectivity_per_unit: 4,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.1,
            noise_phase: 0.05,
            global_inhibition: 0.02,
            hebb_rate: 0.01,
            forget_rate: 0.001,
            prune_below: 0.0001,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.3,
            seed: Some(123),
            causal_decay: 0.01,
        };

        let brain = Brain::new(cfg);
        let mut bytes: Vec<u8> = Vec::new();
        brain.save_image_to(&mut bytes).unwrap();

        let mut cursor = std::io::Cursor::new(bytes);
        let loaded = Brain::load_image_from(&mut cursor).unwrap();

        assert_eq!(loaded.cfg.unit_count, brain.cfg.unit_count);
        assert_eq!(
            loaded.cfg.connectivity_per_unit,
            brain.cfg.connectivity_per_unit
        );
        assert_eq!(loaded.units.len(), brain.units.len());
        assert_eq!(loaded.reserved.len(), brain.reserved.len());
        assert_eq!(loaded.learning_enabled.len(), brain.learning_enabled.len());
        assert_eq!(loaded.symbols_rev, brain.symbols_rev);
    }
}
