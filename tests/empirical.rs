//! Empirical checks with fixed seeds: empirical sequence law against direct
//! target sampling, and empirical per-position acceptance against overlap.

mod common {
    pub mod testutil;
}

use std::collections::HashMap;

use exact_draft_sequence_sampler::{Config, MapTable, Sampler};

const EOS: u32 = 99;
const TOKENS: [u32; 3] = [1, 2, 3];

fn target_weights(len: usize) -> Vec<f64> {
    // Non-iid but memoryless given length; sums to one.
    let raw = match len % 3 {
        0 => vec![0.35, 0.30, 0.20, 0.15],
        1 => vec![0.25, 0.45, 0.15, 0.15],
        _ => vec![0.40, 0.20, 0.30, 0.10],
    };
    raw
}

fn draft_weights(len: usize) -> Vec<f64> {
    let raw = match len % 3 {
        0 => vec![0.20, 0.40, 0.25, 0.15],
        1 => vec![0.30, 0.30, 0.25, 0.15],
        _ => vec![0.35, 0.25, 0.25, 0.15],
    };
    raw
}

fn rows_for(
    max_len: usize,
    weights: impl Fn(usize) -> Vec<f64>,
) -> Vec<(Vec<u32>, Vec<(u32, f64)>)> {
    let mut rows = Vec::new();
    fn rec(
        prefix: Vec<u32>,
        depth: usize,
        max_len: usize,
        rows: &mut Vec<(Vec<u32>, Vec<(u32, f64)>)>,
        weights: &impl Fn(usize) -> Vec<f64>,
    ) {
        if depth > max_len {
            return;
        }
        let w = weights(depth);
        let entries: Vec<(u32, f64)> = TOKENS
            .iter()
            .copied()
            .chain(std::iter::once(EOS))
            .zip(w)
            .collect();
        rows.push((prefix.clone(), entries));
        if depth == max_len {
            return;
        }
        for v in TOKENS {
            let mut p = prefix.clone();
            p.push(v);
            rec(p, depth + 1, max_len, rows, weights);
        }
    }
    rec(vec![], 0, max_len - 1, &mut rows, &weights);
    rows
}

#[test]
fn speculative_sequence_law_matches_direct_target_sampling() {
    let max_len = 3;
    let trials = 200_000;

    // Direct baseline: draft table == target table, window 1.
    let direct_rows = rows_for(max_len, target_weights);
    let spec_target_rows = rows_for(max_len, target_weights);
    let spec_draft_rows = rows_for(max_len, draft_weights);

    let mut direct = HashMap::<Vec<u32>, usize>::new();
    let mut spec = HashMap::<Vec<u32>, usize>::new();

    let direct_t = MapTable::new(direct_rows.clone(), 1e-9).unwrap();
    let direct_d = MapTable::new(direct_rows, 1e-9).unwrap();
    for s in 0..trials as u64 {
        let mut sampler = Sampler::new(
            direct_t.clone(),
            direct_d.clone(),
            Config::new(1, max_len, EOS).unwrap(),
            s.wrapping_mul(7).wrapping_add(1),
        );
        while !sampler.is_finished() {
            sampler.step().unwrap();
        }
        *direct.entry(sampler.sequence().to_vec()).or_default() += 1;
    }

    let spec_t = MapTable::new(spec_target_rows, 1e-9).unwrap();
    let spec_d = MapTable::new(spec_draft_rows, 1e-9).unwrap();
    for s in 0..trials as u64 {
        let mut sampler = Sampler::new(
            spec_t.clone(),
            spec_d.clone(),
            Config::new(2, max_len, EOS).unwrap(),
            s.wrapping_mul(13).wrapping_add(5),
        );
        while !sampler.is_finished() {
            sampler.step().unwrap();
        }
        *spec.entry(sampler.sequence().to_vec()).or_default() += 1;
    }

    let keys: std::collections::BTreeSet<Vec<u32>> =
        direct.keys().chain(spec.keys()).cloned().collect();
    assert!(keys.len() > 20, "expected many distinct sequences, got {}", keys.len());
    let mut max_dev = 0.0f64;
    for k in keys {
        let a = *direct.get(&k).unwrap_or(&0) as f64 / trials as f64;
        let b = *spec.get(&k).unwrap_or(&0) as f64 / trials as f64;
        max_dev = max_dev.max((a - b).abs());
    }
    assert!(
        max_dev < 0.012,
        "sequence law deviates from direct target sampling: {max_dev}"
    );
}

#[test]
fn empirical_acceptance_matches_overlap_mass() {
    let trials = 300_000;
    let mut accepted = 0usize;
    let p_rows = vec![(vec![], vec![(1, 0.50), (2, 0.30), (3, 0.20)])];
    let q_rows = vec![(vec![], vec![(1, 0.20), (2, 0.55), (3, 0.25)])];
    let target = MapTable::new(p_rows, 1e-9).unwrap();
    let draft = MapTable::new(q_rows, 1e-9).unwrap();
    for s in 0..trials as u64 {
        let mut sampler = Sampler::new(
            target.clone(),
            draft.clone(),
            Config::new(1, 1, EOS).unwrap(),
            s.wrapping_mul(1_000_003),
        );
        let r = sampler.step().unwrap();
        if r.accepted == 1 {
            accepted += 1;
        }
    }
    // overlap = sum min(p,q) = min(.5,.2)+min(.3,.55)+min(.2,.25)=.2+.3+.2=.7
    let overlap = 0.7_f64;
    let emp = accepted as f64 / trials as f64;
    assert!(
        (emp - overlap).abs() < 0.004,
        "empirical acceptance {emp} != overlap {overlap}"
    );

    // Identical distributions: always accepted, never repaired.
    let mut repairs = 0;
    let rows = vec![(vec![], vec![(1, 0.50), (2, 0.30), (3, 0.20)])];
    let t_id = MapTable::new(rows.clone(), 1e-9).unwrap();
    let d_id = MapTable::new(rows, 1e-9).unwrap();
    for s in 0..50_000u64 {
        let mut sampler = Sampler::new(
            t_id.clone(),
            d_id.clone(),
            Config::new(1, 1, EOS).unwrap(),
            s.wrapping_add(11),
        );
        let r = sampler.step().unwrap();
        assert_eq!(r.accepted, 1);
        if r.repaired_with.is_some() {
            repairs += 1;
        }
    }
    assert_eq!(repairs, 0, "identical target/draft must never repair");

    // Disjoint support: acceptance zero, repair distribution equals target.
    let p_rows = vec![(vec![], vec![(1, 0.6), (2, 0.4)])];
    let q_rows = vec![(vec![], vec![(3, 0.5), (4, 0.5)])];
    let target = MapTable::new(p_rows, 1e-9).unwrap();
    let draft = MapTable::new(q_rows, 1e-9).unwrap();
    let mut counts = HashMap::<u32, usize>::new();
    for s in 0..200_000u64 {
        let mut sampler = Sampler::new(
            target.clone(),
            draft.clone(),
            Config::new(1, 1, EOS).unwrap(),
            s.wrapping_mul(999_983).wrapping_add(1),
        );
        let r = sampler.step().unwrap();
        assert_eq!(r.accepted, 0);
        assert!(r.rejected_at == Some(0));
        let repaired = r.repaired_with.unwrap();
        *counts.entry(repaired).or_default() += 1;
    }
    let n = 200_000.0;
    let f1 = counts[&1] as f64 / n;
    let f2 = counts[&2] as f64 / n;
    assert!((f1 - 0.6).abs() < 0.006, "disjoint repair token1 {f1}");
    assert!((f2 - 0.4).abs() < 0.006, "disjoint repair token2 {f2}");
}

#[test]
fn zero_probability_tokens_are_never_emitted() {
    let rows = vec![(
        vec![],
        vec![(1, 0.0), (2, 0.7), (3, 0.0), (4, 0.3)],
    )];
    let t = MapTable::new(rows.clone(), 1e-9).unwrap();
    let d = MapTable::new(rows, 1e-9).unwrap();
    for s in 0..20_000u64 {
        let mut sampler =
            Sampler::new(t.clone(), d.clone(), Config::new(1, 1, 10_000).unwrap(), s);
        let r = sampler.step().unwrap();
        let token = r.committed[0];
        assert!(token == 2 || token == 4, "zero-mass token emitted: {token}");
    }
}
