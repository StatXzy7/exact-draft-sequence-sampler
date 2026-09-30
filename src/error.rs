//! Error type covering every explicit failure mode required by the contract.

use std::fmt;

/// Alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SampleError>;

/// All errors are fatal to the current round: a failed `step` / restore never
/// mutates the already committed sequence or checkpoint state.
#[derive(Debug, PartialEq)]
pub enum SampleError {
    /// One of the probabilities was negative.
    NegativeProbability {
        token: u32,
        value: f64,
    },
    /// One of the probabilities was NaN or (plus/minus) infinity.
    NonFiniteProbability {
        token: u32,
        value: f64,
    },
    /// All entries are zero, or the total mass is outside
    /// `1 ± tolerance` (or is not finite).
    InvalidTotalMass {
        prefix: Vec<u32>,
        total: f64,
        tolerance: f64,
    },
    /// The conditional table contains no distribution for this prefix.
    MissingPrefix {
        prefix: Vec<u32>,
    },
    /// A categorical draw is requested from a distribution with no positive
    /// mass (always indicates a bug or a corrupted/invalid table).
    EmptySupport,
    /// Rejection residual has zero mass (mathematically impossible unless the
    /// two supports are disjoint; in that case direct target sampling is used
    /// instead, so reaching this branch is itself an error).
    ZeroResidualMass,
    /// Configuration field out of range.
    InvalidConfig {
        field: &'static str,
        reason: String,
    },
    /// Checkpoint file cannot be parsed.
    CorruptCheckpoint {
        detail: String,
    },
    /// Checkpoint exists but was produced for different configuration / tables
    /// than the ones supplied at resume.
    CheckpointMismatch {
        detail: String,
    },
    /// Checkpoint file could not be read or written.
    CheckpointIo {
        path: String,
        detail: String,
    },
}

impl fmt::Display for SampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SampleError::NegativeProbability { token, value } => write!(
                f,
                "negative probability for token {token}: {value}"
            ),
            SampleError::NonFiniteProbability { token, value } => write!(
                f,
                "non-finite probability for token {token}: {value}"
            ),
            SampleError::InvalidTotalMass { prefix, total, tolerance } => write!(
                f,
                "invalid total mass {total} (tolerance {tolerance}) at prefix {prefix:?}"
            ),
            SampleError::MissingPrefix { prefix } => {
                write!(f, "no conditional distribution for prefix {prefix:?}")
            }
            SampleError::EmptySupport => {
                write!(f, "cannot sample: distribution has no positive-probability token")
            }
            SampleError::ZeroResidualMass => {
                write!(f, "rejection residual has zero mass")
            }
            SampleError::InvalidConfig { field, reason } => {
                write!(f, "invalid config field `{field}`: {reason}")
            }
            SampleError::CorruptCheckpoint { detail } => {
                write!(f, "corrupt checkpoint: {detail}")
            }
            SampleError::CheckpointMismatch { detail } => {
                write!(f, "checkpoint does not match configuration/tables: {detail}")
            }
            SampleError::CheckpointIo { path, detail } => {
                write!(f, "checkpoint I/O error for {path}: {detail}")
            }
        }
    }
}

impl std::error::Error for SampleError {}
