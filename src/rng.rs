//! Deterministic, checkpointable pseudo random source.
//!
//! We use SplitMix64 to derive the full 256-bit internal state of a
//! xoshiro256** generator from a single `u64` seed. xoshiro256** gives
//! high-quality uniform bits while letting a checkpoint save and restore the
//! exact four-word internal state (merely remembering the original seed is
//! therefore not enough to impersonate a resume).

const MASK64: u64 = (1u64 << 53) - 1;

/// Splittable SplitMix64, used only to seed the main generator.
pub fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** (Blackman & Vigna). The four `u64` words are the full state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Derive a generator deterministically from a user supplied seed.
    pub fn from_seed(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut mix = seed;
        for slot in s.iter_mut() {
            mix = mix.wrapping_add(0x9E37_79B9_7F4A_7C15);
            *slot = splitmix64(mix);
        }
        // The all-zero state is the single degenerate state of xoshiro256**;
        // splitmix64 never emits zero here in practice, but guarantee it.
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    /// Reconstruct a generator from its complete internal state.
    pub fn from_state(s: [u64; 4]) -> Self {
        Self { s }
    }

    /// Expose the complete internal state for checkpoint files.
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    fn rotl(x: u64, k: u32) -> u64 {
        x.rotate_left(k)
    }

    /// Raw next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = Self::rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = Self::rotl(self.s[3], 45);
        result
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of mantissa.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() & MASK64) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence() {
        let mut a = Xoshiro256::from_seed(42);
        let mut b = Xoshiro256::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Xoshiro256::from_seed(43);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn state_restore_reproduces_stream() {
        let mut rng = Xoshiro256::from_seed(7);
        for _ in 0..10 {
            rng.next_f64();
        }
        let saved = rng.state();
        let expected: Vec<u64> = (0..20).map(|_| rng.next_u64()).collect();
        let mut restored = Xoshiro256::from_state(saved);
        let actual: Vec<u64> = (0..20).map(|_| restored.next_u64()).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn floats_are_half_open_unit_interval() {
        let mut rng = Xoshiro256::from_seed(1);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}

/// Abstraction over the stream of uniform numbers. The production
/// implementation is [`Xoshiro256`]; tests inject scripted streams to make
/// accept/reject events deterministic.
pub trait UniformSource {
    /// One independent uniform draw in `[0, 1)`.
    fn uniform(&mut self) -> f64;
}

impl UniformSource for Xoshiro256 {
    fn uniform(&mut self) -> f64 {
        self.next_f64()
    }
}
