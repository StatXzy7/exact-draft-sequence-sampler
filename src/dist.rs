//! Finite conditional probability mass functions: validation, categorical
//! sampling and target/draft arithmetic.

use crate::error::{Result, SampleError};
use crate::rng::UniformSource;

/// A finite distribution over token ids, stored as `(token, probability)`
/// pairs. Zero probability entries are legal as long as the *total* mass is
/// one; they can never be returned by sampling.
#[derive(Clone, Debug)]
pub struct Distribution {
    entries: Vec<(u32, f64)>,
}

impl Distribution {
    /// Build and validate a distribution.
    ///
    /// * every probability must be finite and non-negative;
    /// * duplicate token ids are rejected;
    /// * the total mass must be within `tol` of `1.0`;
    /// * at least one entry must have strictly positive mass.
    pub fn new(entries: Vec<(u32, f64)>, tol: f64) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        let mut total = 0.0f64;
        for (token, p) in &entries {
            if p.is_nan() || p.is_infinite() {
                return Err(SampleError::NonFiniteProbability {
                    token: *token,
                    value: *p,
                });
            }
            if *p < 0.0 {
                return Err(SampleError::NegativeProbability {
                    token: *token,
                    value: *p,
                });
            }
            if !seen.insert(*token) {
                return Err(SampleError::InvalidConfig {
                    field: "distribution",
                    reason: format!("duplicate token {token}"),
                });
            }
            total += *p;
        }
        if !total.is_finite() || (total - 1.0).abs() > tol {
            return Err(SampleError::InvalidTotalMass {
                prefix: Vec::new(),
                total,
                tolerance: tol,
            });
        }
        if entries.iter().all(|(_, p)| *p == 0.0) {
            return Err(SampleError::EmptySupport);
        }
        Ok(Self { entries })
    }

    /// Probability of `token` (zero when absent or explicitly zero).
    pub fn mass(&self, token: u32) -> f64 {
        self.entries
            .iter()
            .find(|(t, _)| *t == token)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }

    /// Iterate `(token, probability)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.entries.iter().copied()
    }

    /// Inverse-CDF categorical draw. Only entries with strictly positive mass
    /// can ever be selected; the draw is normalized by the stored total so the
    /// returned token law is exactly the supplied probabilities (up to the
    /// unavoidable 2^-53 discretization of a uniform `f64`).
    pub fn draw(&self, rng: &mut impl UniformSource) -> Result<u32> {
        let total: f64 = self.entries.iter().map(|(_, p)| *p).sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(SampleError::EmptySupport);
        }
        let u = rng.uniform() * total;
        let mut acc = 0.0f64;
        let mut last = None;
        for (token, p) in self.entries.iter().copied() {
            if p > 0.0 {
                last = Some(token);
                acc += p;
                if u < acc {
                    return Ok(token);
                }
            }
        }
        // Floating-point guard: return the last positive-mass entry instead of
        // a zero-probability one.
        Ok(last.expect("non-empty support guaranteed by validation"))
    }
}

/// Acceptance probability for a draft token: `min(1, p(x)/q(x))`.
///
/// Tokens outside the draft support (`q == 0`) are always rejected. Tokens
/// outside the target support (`p == 0`, `q > 0`) are always rejected too.
pub fn acceptance_prob(p: f64, q: f64) -> f64 {
    if q <= 0.0 {
        0.0
    } else if p >= q {
        1.0
    } else {
        p / q
    }
}

/// Normalized rejection residual for position i:
/// `(p_i(x) - q_i(x))_+ / sum_y (p_i(y) - q_i(y))_+`.
///
/// Returns `None` when the residual mass is (numerically) zero, which happens
/// exactly when `p_i` is dominated by `q_i` everywhere — in particular when
/// target and draft distributions are identical. In that situation a rejection
/// event has probability zero and must never be repaired through this branch.
pub fn residual_distribution(
    target: &Distribution,
    draft: &Distribution,
    tol: f64,
) -> Result<Option<Distribution>> {
    let mut residual: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    for (token, p) in target.iter() {
        let q = draft.mass(token);
        if p > q {
            residual.insert(token, p - q);
        }
    }
    let total: f64 = residual.values().sum();
    if total <= tol {
        return Ok(None);
    }
    let entries: Vec<(u32, f64)> = residual
        .into_iter()
        .map(|(token, w)| (token, w / total))
        .collect();
    Ok(Some(Distribution::new(entries, tol)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256;

    #[test]
    fn rejects_illegal_distributions() {
        assert!(matches!(
            Distribution::new(vec![(1, f64::NAN)], 1e-9),
            Err(SampleError::NonFiniteProbability { .. })
        ));
        assert!(matches!(
            Distribution::new(vec![(1, f64::INFINITY)], 1e-9),
            Err(SampleError::NonFiniteProbability { .. })
        ));
        assert!(matches!(
            Distribution::new(vec![(1, -0.1), (2, 1.1)], 1e-9),
            Err(SampleError::NegativeProbability { .. })
        ));
        assert!(matches!(
            Distribution::new(vec![(1, 0.4), (2, 0.5)], 1e-9),
            Err(SampleError::InvalidTotalMass { .. })
        ));
        assert!(matches!(
            Distribution::new(vec![(1, 0.0), (2, 0.0), (3, 1.0 - 0.0)], 1e-9)
                .map(|d| d.entries.len()),
            Ok(3)
        ));
    }

    #[test]
    fn zero_mass_tokens_are_never_drawn() {
        let d = Distribution::new(
            vec![(1, 0.0), (2, 1.0), (3, 0.0)],
            1e-9,
        )
        .unwrap();
        let mut rng = Xoshiro256::from_seed(0);
        for _ in 0..1000 {
            assert_eq!(d.draw(&mut rng).unwrap(), 2);
        }
    }
}
