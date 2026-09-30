//! Local conditional probability tables for target and draft.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::dist::Distribution;
use crate::error::{Result, SampleError};

/// A locally queryable conditional distribution table.
///
/// Every query must use the *real* committed prefix; the sampler never queries
/// with a speculative (draft) continuation.
pub trait ConditionalTable {
    /// Return the validated distribution for `prefix`, or
    /// [`SampleError::MissingPrefix`] / a validity error.
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution>;

    /// Stable fingerprint of the whole table, used to reject checkpoints saved
    /// against different tables. The default is a conservative constant that
    /// effectively disables this part of the match check; table
    /// implementations should override it (as [`MapTable`] does).
    fn fingerprint(&self) -> u64 {
        0
    }
}

/// In-memory implementation keyed by exact prefix sequences. The constructor
/// validates every distribution up front, so a round cannot partially consume a
/// bad table.
#[derive(Clone)]
pub struct MapTable {
    inner: Arc<HashMap<Vec<u32>, Vec<(u32, f64)>>>,
    fingerprint: u64,
    tol: f64,
}

impl MapTable {
    /// Build and fully validate a table. `tol` is the allowed deviation of the
    /// per-prefix total mass from `1.0`.
    pub fn new(
        rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>,
        tol: f64,
    ) -> Result<Self> {
        let mut map = HashMap::new();
        for (prefix, entries) in rows {
            // Validate with the prefix attached to the error.
            let dist = Distribution::new(entries, tol).map_err(|e| match e {
                SampleError::InvalidTotalMass {
                    total, tolerance, ..
                } => SampleError::InvalidTotalMass {
                    prefix: prefix.clone(),
                    total,
                    tolerance,
                },
                other => other,
            })?;
            map.insert(prefix, dist_entries(&dist));
        }
        let fingerprint = fingerprint_map(&map);
        Ok(Self {
            inner: Arc::new(map),
            fingerprint,
            tol,
        })
    }

    /// Mass tolerance configured for this table.
    pub fn tolerance(&self) -> f64 {
        self.tol
    }
}

fn dist_entries(d: &Distribution) -> Vec<(u32, f64)> {
    d.iter().collect()
}

impl ConditionalTable for MapTable {
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution> {
        let entries = self
            .inner
            .get(prefix)
            .ok_or_else(|| SampleError::MissingPrefix {
                prefix: prefix.to_vec(),
            })?
            .clone();
        // Rows were validated at construction; cloning keeps validation honest.
        Distribution::new(entries, self.tol)
    }

    fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// Order-independent FNV-1a-style fingerprint of every prefix/entry pair.
fn fingerprint_map(map: &HashMap<Vec<u32>, Vec<(u32, f64)>>) -> u64 {
    // XOR of per-row hashes makes the fingerprint insensitive to row order,
    // while still changing when any prefix, token or probability changes.
    let mut acc = 0u64;
    for (prefix, entries) in map {
        let mut hasher = Fnv1a::default();
        prefix.hash(&mut hasher);
        entries.len().hash(&mut hasher);
        for (token, p) in entries {
            token.hash(&mut hasher);
            p.to_bits().hash(&mut hasher);
        }
        acc ^= hasher.finish();
    }
    acc
}

struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        Fnv1a(0xCBF2_9CE4_8422_2325)
    }
}

impl Hasher for Fnv1a {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}
