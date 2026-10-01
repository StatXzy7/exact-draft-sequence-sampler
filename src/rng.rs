//! Minimal pluggable randomness.
//!
//! The sampler is generic over [`Rng`] so tests can drive it with a fully
//! scripted stream while production code uses the bundled [`SplitMix64`].

/// A stream of uniform samples in `[0, 1)`.
///
/// The full generator state must fit in 64 bits so that checkpoints and
/// atomic rollback of failed rounds can rely on [`Rng::state`] /
/// [`Rng::set_state`]. Returning exact `0.0` is legal; exact `1.0` must never
/// be produced.
pub trait Rng {
    /// Build a fresh stream from a seed.
    fn from_seed(seed: u64) -> Self;
    /// Next uniform sample.
    fn next_u01(&mut self) -> f64;
    /// Capture the complete current state.
    fn state(&self) -> u64;
    /// Restore a state previously captured with [`Rng::state`].
    fn set_state(&mut self, state: u64);
}

/// Deterministic 64-bit SplitMix64 generator (Sebastiano Vigna).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create from a user supplied seed.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
}

impl Rng for SplitMix64 {
    fn from_seed(seed: u64) -> Self {
        SplitMix64::new(seed)
    }

    fn next_u01(&mut self) -> f64 {
        // SplitMix64 mixing constants.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 53 bits -> uniform f64 in [0, 1).
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn state(&self) -> u64 {
        self.state
    }

    fn set_state(&mut self, state: u64) {
        self.state = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_uniform_range_and_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            let x = a.next_u01();
            assert!((0.0..1.0).contains(&x));
            assert_eq!(x, b.next_u01());
        }
        let mut other = SplitMix64::new(7);
        assert_ne!(other.next_u01(), SplitMix64::new(42).next_u01());
    }

    #[test]
    fn splitmix_state_roundtrip() {
        let mut rng = SplitMix64::new(123);
        for _ in 0..10 {
            rng.next_u01();
        }
        let state = rng.state();
        let before: Vec<f64> = (0..5).map(|_| rng.next_u01()).collect();
        rng.set_state(state);
        let after: Vec<f64> = (0..5).map(|_| rng.next_u01()).collect();
        assert_eq!(before, after);
    }
}
