//! Exact-output-law speculative sequence sampler for finite vocabularies.
//!
//! A lightweight *draft* conditional table proposes up to `window` tokens per
//! round; a *target* conditional table verifies them with the standard
//! acceptance test `min(1, p/q)` and repairs the first rejected position from
//! the normalized residual `(p - q)_+`. The committed sequence has exactly the
//! law of drawing every token directly from the target distribution, while the
//! per-position acceptance probability equals the overlap mass
//! `sum_x min(p(x), q(x))`.
//!
//! No model download and no third-party runtime dependency is required; tables
//! are queried locally against the real committed prefix.

pub mod dist;
pub mod error;
pub mod rng;
pub mod sampler;
pub mod table;

pub use dist::{acceptance_prob, residual_distribution, Distribution};
pub use error::{Result, SampleError};
pub use rng::{UniformSource, Xoshiro256};
pub use sampler::{Checkpoint, Config, RoundReport, Sampler};
pub use table::{ConditionalTable, MapTable};
