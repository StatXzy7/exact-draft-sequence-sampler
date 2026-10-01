//! # Exact draft sequence sampler
//!
//! Speculative sampling over finite-vocabulary conditional probability tables.
//! A lightweight *draft* table proposes a contiguous run of candidate tokens;
//! a *target* table verifies them with the standard accept/reject rule and a
//! normalized-residual correction at the first rejection. The resulting
//! marginal distribution is identical to sampling directly from the target
//! distribution token by token, and the per-position acceptance probability
//! equals the overlap mass `sum_k min(target_k, draft_k)`.
//!
//! Every conditional query is made against the currently committed (real)
//! prefix. Rounds are atomic: a missing or invalid distribution, a negative
//! probability or non-finite value, an illegal total mass, or a bad
//! checkpoint restore all fail without changing committed results or the
//! random stream.
//!
//! See [`Sampler::run_round`] and the `tests/` directory.

mod checkpoint;
mod error;
mod model;
mod rng;
mod sampler;

pub use error::{DistributionError, QueryKind, SamplerError};
pub use model::{validate_distribution, ConditionalTable, Config, HashMapTable, MASS_TOLERANCE};
pub use rng::{Rng, SplitMix64};
pub use sampler::{Acceptance, RoundEvent, Sampler, StopReason};
