// Minimal PRNG (no external crates).
//
// This is NOT cryptographically secure.
// It is used only for controlled noise/exploration and reproducible evaluation.

#[derive(Debug, Clone)]
pub struct Prng {
    state: u64,
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        // Avoid a zero state.
        let seed = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: seed }
    }

    #[cfg(feature = "std")]
    pub(crate) fn from_state(state: u64) -> Self {
        // Avoid a zero state.
        let state = if state == 0 {
            0x9E3779B97F4A7C15
        } else {
            state
        };
        Self { state }
    }

    #[cfg(feature = "std")]
    pub(crate) fn state(&self) -> u64 {
        self.state
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        // Marsaglia / Vigna family. Simple, fast, decent for simulation noise.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    pub fn next_f32_01(&mut self) -> f32 {
        // Convert to [0,1).
        let x = self.next_u32();
        (x as f32) / (u32::MAX as f32 + 1.0)
    }

    #[inline]
    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32_01()
    }

    #[inline]
    pub fn gen_range_usize(&mut self, low: usize, high: usize) -> usize {
        if high <= low {
            return low;
        }
        let span = (high - low) as u32;
        let v = self.next_u32() % span;
        low + v as usize
    }
}
