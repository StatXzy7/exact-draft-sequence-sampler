//! Error types for the exact speculative sampler.

use std::error::Error;
use std::fmt;

/// Why a conditional probability row was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum DistributionError {
    /// The row length does not match the configured vocabulary size.
    VocabSize { expected: usize, found: usize },
    /// At least one probability is negative.
    Negative(usize),
    /// At least one probability is NaN, positive or negative infinity.
    NonFinite(usize),
    /// The row is empty (no support at all).
    Empty,
    /// Total mass is not within tolerance of one.
    TotalMass { total: f64, tolerance: f64 },
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionError::VocabSize { expected, found } => write!(
                f,
                "distribution length {found} does not match vocabulary size {expected}"
            ),
            DistributionError::Negative(i) => write!(f, "negative probability at token {i}"),
            DistributionError::NonFinite(i) => write!(f, "non-finite probability at token {i}"),
            DistributionError::Empty => write!(f, "distribution has no entries"),
            DistributionError::TotalMass { total, tolerance } => write!(
                f,
                "total mass {total} is not within {tolerance} of 1.0"
            ),
        }
    }
}

impl Error for DistributionError {}

/// Which table was queried when something went wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryKind {
    /// Target conditional distribution.
    Target,
    /// Draft conditional distribution.
    Draft,
}

impl QueryKind {
    fn name(self) -> &'static str {
        match self {
            QueryKind::Target => "target",
            QueryKind::Draft => "draft",
        }
    }
}

/// All failure modes exposed by the sampler.
///
/// Every failure returned by [`crate::Sampler::run_round`] is detected before
/// the committed sequence is mutated: on error the sampler keeps producing the
/// same future events as it would have produced without the failed attempt.
#[derive(Clone, PartialEq, Debug)]
pub enum SamplerError {
    /// A configurable parameter was outside its valid range.
    InvalidConfig(String),
    /// The initial prefix contains a token outside the vocabulary.
    InvalidPrefixToken(usize),
    /// No conditional row exists for the queried (real) prefix.
    MissingDistribution {
        kind: QueryKind,
        prefix: Vec<usize>,
    },
    /// The row stored for a prefix was not a legal probability distribution.
    InvalidDistribution {
        kind: QueryKind,
        prefix: Vec<usize>,
        source: DistributionError,
    },
    /// Internal invariant broken: residual mass zero at a rejection position.
    ZeroResidual { prefix: Vec<usize>, token: usize },
    /// A checkpoint buffer could not be decoded or failed its checksum.
    CorruptCheckpoint(String),
    /// A checkpoint was restored with a configuration different from the one
    /// it was created with.
    ConfigMismatch { field: String },
    /// Checkpoint fingerprint differs from the one computed over the tables.
    ModelMismatch,
    /// IO failure while writing or reading a checkpoint file.
    Io(String),
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerError::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            SamplerError::InvalidPrefixToken(t) => {
                write!(f, "prefix token {t} is outside the vocabulary")
            }
            SamplerError::MissingDistribution { kind, prefix } => write!(
                f,
                "{} distribution is missing for prefix {:?}",
                kind.name(),
                prefix
            ),
            SamplerError::InvalidDistribution { kind, prefix, source } => write!(
                f,
                "invalid {} distribution at prefix {:?}: {source}",
                kind.name(),
                prefix
            ),
            SamplerError::ZeroResidual { prefix, token } => write!(
                f,
                "internal error: zero residual mass while correcting token {token} at prefix {:?}",
                prefix
            ),
            SamplerError::CorruptCheckpoint(msg) => {
                write!(f, "corrupt checkpoint: {msg}")
            }
            SamplerError::ConfigMismatch { field } => {
                write!(f, "checkpoint/config mismatch in field '{field}'")
            }
            SamplerError::ModelMismatch => {
                write!(f, "checkpoint model fingerprint does not match the supplied tables")
            }
            SamplerError::Io(msg) => write!(f, "checkpoint IO error: {msg}"),
        }
    }
}

impl Error for SamplerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SamplerError::InvalidDistribution { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SamplerError>;
