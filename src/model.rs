//! Configuration and locally queryable conditional probability tables.

use crate::error::{DistributionError, Result, SamplerError};
use std::collections::HashMap;

/// Maximum allowed deviation of a row's total mass from 1.
pub const MASS_TOLERANCE: f64 = 1e-9;

/// Static configuration of a sampling run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Config {
    /// Number of tokens in the vocabulary; valid ids are `0..vocab_size`.
    pub vocab_size: usize,
    /// Number of draft tokens proposed each round (`>= 1`).
    pub draft_window: usize,
    /// Inclusive upper bound on the committed sequence length.
    pub max_length: usize,
    /// Token id that terminates generation as soon as it is committed.
    pub eos_token: usize,
    /// Initial random seed.
    pub seed: u64,
}

impl Config {
    /// Construct and validate a configuration.
    pub fn new(
        vocab_size: usize,
        draft_window: usize,
        max_length: usize,
        eos_token: usize,
        seed: u64,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(SamplerError::InvalidConfig(
                "vocab_size must be positive".into(),
            ));
        }
        if draft_window == 0 {
            return Err(SamplerError::InvalidConfig(
                "draft_window must be positive".into(),
            ));
        }
        if eos_token >= vocab_size {
            return Err(SamplerError::InvalidConfig(format!(
                "eos_token {eos_token} is outside vocabulary of size {vocab_size}"
            )));
        }
        if max_length == 0 {
            return Err(SamplerError::InvalidConfig(
                "max_length must be positive".into(),
            ));
        }
        Ok(Config {
            vocab_size,
            draft_window,
            max_length,
            eos_token,
            seed,
        })
    }
}

/// Check that `row` is a probability distribution of `vocab_size` categories.
pub fn validate_distribution(row: &[f64], vocab_size: usize) -> std::result::Result<(), DistributionError> {
    if row.is_empty() {
        return Err(DistributionError::Empty);
    }
    if row.len() != vocab_size {
        return Err(DistributionError::VocabSize {
            expected: vocab_size,
            found: row.len(),
        });
    }
    let mut total = 0.0f64;
    for (i, &p) in row.iter().enumerate() {
        if p.is_nan() || p.is_infinite() {
            return Err(DistributionError::NonFinite(i));
        }
        if p < 0.0 {
            return Err(DistributionError::Negative(i));
        }
        total += p;
    }
    if (total - 1.0).abs() > MASS_TOLERANCE {
        return Err(DistributionError::TotalMass {
            total,
            tolerance: MASS_TOLERANCE,
        });
    }
    Ok(())
}

/// A locally queryable table of conditional token distributions.
///
/// Every query is conditional on the *real* prefix currently committed: the
/// sampler only calls this method with a prefix consisting of tokens it has
/// already accepted, never with speculative tokens.
pub trait ConditionalTable {
    /// Return `P(token | prefix)`. Returning `Ok(row)` with a row that fails
    /// validation, or `Err(MissingDistribution)`, both abort the round without
    /// changing the committed sequence.
    fn conditional(&self, prefix: &[usize]) -> Result<Vec<f64>>;

    /// Stable fingerprint over all table entries.
    ///
    /// It is embedded in checkpoints so that restoring with different tables
    /// fails explicitly instead of silently continuing.
    fn fingerprint(&self) -> u64;
}

/// Simple `HashMap`-backed conditional table.
#[derive(Clone, Debug, Default)]
pub struct HashMapTable {
    vocab_size: usize,
    rows: HashMap<Vec<usize>, Vec<f64>>,
}

impl HashMapTable {
    /// Create an empty table for `vocab_size` token categories.
    pub fn new(vocab_size: usize) -> Self {
        HashMapTable {
            vocab_size,
            rows: HashMap::new(),
        }
    }

    /// Insert or replace the distribution for `prefix`; the row is validated.
    pub fn insert(&mut self, prefix: Vec<usize>, row: Vec<f64>) -> Result<()> {
        validate_distribution(&row, self.vocab_size).map_err(|source| {
            SamplerError::InvalidDistribution {
                kind: crate::error::QueryKind::Target,
                prefix: prefix.clone(),
                source,
            }
        })?;
        self.rows.insert(prefix, row);
        Ok(())
    }

    /// Number of stored prefixes.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table stores no prefixes.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl ConditionalTable for HashMapTable {
    fn conditional(&self, prefix: &[usize]) -> Result<Vec<f64>> {
        self.rows
            .get(prefix)
            .cloned()
            .ok_or_else(|| SamplerError::MissingDistribution {
                kind: crate::error::QueryKind::Target,
                prefix: prefix.to_vec(),
            })
    }

    fn fingerprint(&self) -> u64 {
        let mut entries: Vec<(&Vec<usize>, &Vec<f64>)> = self.rows.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut hash = fnv1a_init();
        for (prefix, row) in entries {
            for &token in prefix {
                hash = fnv1a_u64(hash, token as u64);
            }
            hash = fnv1a_u64(hash, u64::MAX); // prefix separator
            for &p in row {
                fnv1a_bytes(&mut hash, &p.to_le_bytes());
            }
            hash = fnv1a_u64(hash, u64::MAX); // row separator
        }
        hash
    }
}

pub(crate) fn fnv1a_init() -> u64 {
    0xcbf2_9ce4_8422_2325
}

pub(crate) fn fnv1a_u64(mut hash: u64, value: u64) -> u64 {
    fnv1a_bytes(&mut hash, &value.to_le_bytes());
    hash
}

pub(crate) fn fnv1a_bytes(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(Config::new(0, 1, 2, 0, 1).is_err());
        assert!(Config::new(2, 0, 2, 0, 1).is_err());
        assert!(Config::new(2, 1, 0, 0, 1).is_err());
        assert!(Config::new(2, 1, 2, 2, 1).is_err());
        assert!(Config::new(2, 1, 2, 1, 1).is_ok());
    }

    #[test]
    fn row_validation_cases() {
        assert!(validate_distribution(&[], 2).is_err());
        assert!(validate_distribution(&[0.5], 2).is_err());
        assert!(validate_distribution(&[-0.5, 1.5], 2).is_err());
        assert!(validate_distribution(&[f64::NAN, 1.0], 2).is_err());
        assert!(validate_distribution(&[f64::INFINITY, 0.0], 2).is_err());
        assert!(validate_distribution(&[0.3, 0.3], 2).is_err());
        assert!(validate_distribution(&[0.0, 1.0], 2).is_ok());
        assert!(validate_distribution(&[0.3, 0.7000000001], 2).is_ok());
    }

    #[test]
    fn fingerprints_are_stable_and_sensitive() {
        let mut a = HashMapTable::new(2);
        a.insert(vec![], vec![0.4, 0.6]).unwrap();
        let mut b = HashMapTable::new(2);
        b.insert(vec![], vec![0.4, 0.6]).unwrap();
        assert_eq!(a.fingerprint(), b.fingerprint());
        let mut c = HashMapTable::new(2);
        c.insert(vec![], vec![0.6, 0.4]).unwrap();
        assert_ne!(a.fingerprint(), c.fingerprint());
    }
}
