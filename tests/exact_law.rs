//! Algebraic checks that the committed token law is exactly the target law and
//! that first-position acceptance probability equals overlap mass.
//!
//! All probabilities are integer numerators over a shared `DENOM`, so the
//! identity is checked without sampling noise and independent of the window
//! size (the marginal law of the first committed token does not depend on how
//! many later draft tokens are proposed).

mod common {
    pub mod testutil;
}

#[derive(Clone)]
struct Law {
    entries: Vec<(u32, i128)>, // token -> numerator over DENOM
}

const DENOM: i128 = 600; // divisible by 2,3,4,5,6,10,12

impl Law {
    fn new(raw: &[(u32, i128, i128)]) -> Self {
        let entries: Vec<(u32, i128)> = raw
            .iter()
            .map(|(t, n, d)| (*t, n * DENOM / d))
            .collect();
        let total: i128 = entries.iter().map(|(_, n)| n).sum();
        assert_eq!(total, DENOM, "law must sum to 1");
        Self { entries }
    }
    fn num(&self, token: u32) -> i128 {
        self.entries
            .iter()
            .find(|(t, _)| *t == token)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    }
    fn tokens(&self) -> Vec<u32> {
        self.entries.iter().map(|(t, _)| *t).collect()
    }
}

#[allow(clippy::type_complexity)]
fn cases() -> Vec<(Law, Law)> {
    vec![
        // identical
        (
            Law::new(&[(1, 1, 10), (2, 2, 10), (3, 3, 10), (4, 4, 10)]),
            Law::new(&[(1, 1, 10), (2, 2, 10), (3, 3, 10), (4, 4, 10)]),
        ),
        // disjoint supports
        (
            Law::new(&[(1, 3, 10), (2, 7, 10)]),
            Law::new(&[(3, 2, 10), (4, 8, 10)]),
        ),
        // partial overlap
        (
            Law::new(&[(1, 2, 10), (2, 3, 10), (3, 5, 10)]),
            Law::new(&[(1, 4, 10), (2, 1, 10), (3, 5, 10)]),
        ),
        // target support inside draft support
        (
            Law::new(&[(1, 6, 10), (2, 4, 10)]),
            Law::new(&[(1, 3, 10), (2, 3, 10), (3, 4, 10)]),
        ),
        // draft support inside target support
        (
            Law::new(&[(1, 3, 10), (2, 3, 10), (3, 4, 10)]),
            Law::new(&[(1, 6, 10), (2, 4, 10)]),
        ),
    ]
}

#[test]
fn first_token_output_law_equals_target_and_acceptance_equals_overlap() {
    for (p, q) in cases() {
        // Use exact rational accumulation in f64 only after integer math per
        // contribution; all chosen cases divide evenly.
        let support: std::collections::BTreeSet<u32> =
            p.tokens().into_iter().chain(q.tokens()).collect();
        let residual_total: f64 = support
            .iter()
            .map(|y| (p.num(*y) - q.num(*y)).max(0) as f64)
            .sum::<f64>();

        let mut mass = std::collections::BTreeMap::<u32, f64>::new();
        let mut overlap = 0.0f64;
        for z in support.iter() {
            let pz = p.num(*z) as f64;
            let qz = q.num(*z) as f64;
            overlap += pz.min(qz) / DENOM as f64;
            // accepted via proposal z
            *mass.entry(*z).or_default() += pz.min(qz) / DENOM as f64;
            // rejected proposals repaired into x
            if qz > pz {
                let reject = (qz - pz) / DENOM as f64;
                for x in support.iter() {
                    let w = (p.num(*x) - q.num(*x)).max(0) as f64 / DENOM as f64;
                    if w > 0.0 {
                        *mass.entry(*x).or_default() +=
                            reject * (w / (residual_total / DENOM as f64));
                    }
                }
            }
        }

        let mut total = 0.0f64;
        for x in support.iter() {
            let expected = p.num(*x) as f64 / DENOM as f64;
            let got = mass.get(x).copied().unwrap_or(0.0);
            assert!(
                (got - expected).abs() < 1e-12,
                "token {x}: output mass {got} != target {expected}, p={:?} q={:?}",
                p.entries,
                q.entries
            );
            total += got;
        }
        assert!((total - 1.0).abs() < 1e-12, "total mass {total}");

        // Acceptance: sum_z q(z)*min(1,p/q) = sum min = overlap, and it must
        // lie in [0,1]; identical -> 1, disjoint -> 0.
        assert!((0.0..=1.0 + 1e-12).contains(&overlap));
        let disjoint = support
            .iter()
            .all(|t| p.num(*t) == 0 || q.num(*t) == 0);
        if disjoint {
            assert!(overlap.abs() < 1e-12);
        }
        let identical = support
            .iter()
            .all(|t| p.num(*t) == q.num(*t));
        if identical {
            assert!((overlap - 1.0).abs() < 1e-12);
        }
    }
}
