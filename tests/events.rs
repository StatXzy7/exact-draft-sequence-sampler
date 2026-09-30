//! Deterministic event assertions using a scripted uniform stream:
//! first-rejection cutoff, EOS immediacy, length boundary, disjoint support.

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::testutil::{RecordingTable, ScriptedRng, Which};
use exact_draft_sequence_sampler::{Config, Sampler};

const EOS: u32 = 0;

type Rows = Vec<(Vec<u32>, Vec<(u32, f64)>)>;

fn rows_at(prefixes: &[Vec<u32>], entries: Vec<(u32, f64)>) -> Rows {
    prefixes.iter().map(|p| (p.clone(), entries.clone())).collect()
}

fn prefixes_up_to(len: usize) -> Vec<Vec<u32>> {
    let mut out = vec![vec![]];
    let mut frontier = vec![vec![]];
    for _ in 0..len {
        let mut next = Vec::new();
        for p in frontier {
            for t in [1u32, 2, 3] {
                let mut q = p.clone();
                q.push(t);
                next.push(q.clone());
                out.push(q);
            }
        }
        frontier = next;
    }
    out
}

fn target_entries() -> Vec<(u32, f64)> {
    vec![(EOS, 0.1), (1, 0.5), (2, 0.2), (3, 0.2)]
}

fn assert_queries(
    log: &Rc<RefCell<Vec<(Which, Vec<u32>)>>>,
    expected: &[(Which, Vec<u32>)],
) {
    let got = log.borrow().clone();
    assert_eq!(got, expected.to_vec(), "query log mismatch");
}

#[test]
fn first_rejection_cuts_off_and_discards_tail() {
    let pref = prefixes_up_to(3);
    // Draft proposes token 2 deterministically everywhere.
    let (target, draft, log) = RecordingTable::pair(
        rows_at(&pref, target_entries()),
        rows_at(&pref, vec![(2, 1.0)]),
    );
    // pos0: draft draw u=.5 -> token 2; accept test u=.9 vs min(1,.2)=.2 reject
    // residual at []: weights eos .1, 1 .5, 3 .2 (norm .8); u=.05 -> eos
    let rng = ScriptedRng::new(&[0.5, 0.9, 0.05]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(3, 8, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();

    assert_eq!(r.proposed, vec![2]);
    assert_eq!(r.accepted, 0);
    assert_eq!(r.rejected_at, Some(0));
    assert_eq!(r.repaired_with, Some(EOS));
    assert_eq!(r.committed, vec![EOS]);
    assert!(r.finished);
    assert_queries(
        &log,
        &[(Which::Draft, vec![]), (Which::Target, vec![])],
    );
}

#[test]
fn accepted_then_rejected_continues_only_on_true_prefix() {
    let pref = prefixes_up_to(3);
    let (target, draft, log) = RecordingTable::pair(
        rows_at(&pref, target_entries()),
        rows_at(&pref, vec![(2, 1.0)]),
    );
    // pos0 draft 2 accept (u=.1<.2); pos1 at [2]: draft 2 reject (u=.99);
    // residual draw u=.5 -> weights eos .125, 1 .625 cumul .125,.75 -> token1
    let rng = ScriptedRng::new(&[0.5, 0.1, 0.5, 0.99, 0.5]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(3, 8, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();

    assert_eq!(r.proposed, vec![2, 2]);
    assert_eq!(r.accepted, 1);
    assert_eq!(r.rejected_at, Some(1));
    assert_eq!(r.rejected_prefix(), Some(vec![2]));
    assert_eq!(r.repaired_with, Some(1));
    assert_eq!(r.committed, vec![2, 1]);
    assert_eq!(r.sequence, vec![2, 1]);
    assert_queries(
        &log,
        &[
            (Which::Draft, vec![]),
            (Which::Target, vec![]),
            (Which::Draft, vec![2]),
            (Which::Target, vec![2]),
        ],
    );
}

#[test]
fn accepting_eos_stops_round_immediately_with_window_remaining() {
    let (target, draft, log) = RecordingTable::pair(
        vec![(vec![], vec![(EOS, 0.5), (1, 0.5)])],
        vec![(vec![], vec![(EOS, 1.0)])],
    );
    // draft draw 0 -> eos; accept test 0 -> accept; stop at once
    let rng = ScriptedRng::new(&[0.0, 0.0]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(4, 8, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();
    assert_eq!(r.proposed, vec![EOS]);
    assert_eq!(r.committed, vec![EOS]);
    assert_eq!(r.accepted, 1);
    assert!(r.bonus_token.is_none());
    assert!(r.finished);
    assert_queries(
        &log,
        &[(Which::Draft, vec![]), (Which::Target, vec![])],
    );
}

#[test]
fn length_boundary_suppresses_bonus_and_excess_proposals() {
    let pref = vec![vec![], vec![1]];
    let (target, draft, log) = RecordingTable::pair(
        rows_at(&pref, vec![(1, 1.0)]),
        rows_at(&pref, vec![(1, 1.0)]),
    );
    // room = 1: draft 1 accept; bonus suppressed because no room
    let rng = ScriptedRng::new(&[0.0, 0.0]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(3, 1, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();
    assert_eq!(r.committed, vec![1]);
    assert!(r.bonus_token.is_none());
    assert_eq!(r.proposed, vec![1]);
    assert!(r.finished);
    let r2 = sampler.step().unwrap();
    assert!(r2.committed.is_empty() && r2.finished);
    assert_queries(
        &log,
        &[(Which::Draft, vec![]), (Which::Target, vec![])],
    );
}

#[test]
fn repair_fills_last_slot_exactly_to_cap() {
    let (target, draft, log) = RecordingTable::pair(
        vec![
            (vec![], vec![(1, 1.0)]),
            (vec![1], vec![(1, 0.2), (2, 0.8)]),
        ],
        vec![
            (vec![], vec![(1, 1.0)]),
            (vec![1], vec![(2, 1.0)]),
        ],
    );
    // pos0 draft1 accept; pos1 draft2 reject (.9>=.8); residual at [1] puts
    // all mass on token 1 -> repair 1
    let rng = ScriptedRng::new(&[0.0, 0.0, 0.0, 0.9, 0.0]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(2, 2, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();
    assert_eq!(r.proposed, vec![1, 2]);
    assert_eq!(r.accepted, 1);
    assert_eq!(r.rejected_at, Some(1));
    assert_eq!(r.repaired_with, Some(1));
    assert_eq!(r.committed, vec![1, 1]);
    assert_eq!(sampler.sequence(), vec![1, 1]);
    assert!(r.finished);
    assert_queries(
        &log,
        &[
            (Which::Draft, vec![]),
            (Which::Target, vec![]),
            (Which::Draft, vec![1]),
            (Which::Target, vec![1]),
        ],
    );
}

#[test]
fn disjoint_support_repair_target_law_and_never_accepts() {
    let (target, draft, log) = RecordingTable::pair(
        vec![(vec![], vec![(1, 0.6), (2, 0.4)])],
        vec![(vec![], vec![(3, 0.5), (4, 0.5)])],
    );
    // draft draw .0 -> token3; acceptance min(1,.0/.5)=0 -> reject; residual
    // equals the target distribution; u=.2 -> token1
    let rng = ScriptedRng::new(&[0.0, 0.5, 0.2]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(2, 8, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();
    assert_eq!(r.accepted, 0);
    assert_eq!(r.repaired_with, Some(1));
    assert_eq!(r.committed, vec![1]);
    assert_queries(
        &log,
        &[(Which::Draft, vec![]), (Which::Target, vec![])],
    );
}

#[test]
fn full_window_accepted_samples_one_bonus_target_token() {
    let pref = vec![vec![], vec![1], vec![1, 1]];
    let (target, draft, _log) = RecordingTable::pair(
        rows_at(&pref, vec![(1, 0.9), (2, 0.1)]),
        rows_at(&pref, vec![(1, 1.0)]),
    );
    // window 2, max_len 8: accept 1 at [] and [1]; bonus draw at [1,1] u=0.95
    // -> token 2
    let rng = ScriptedRng::new(&[0.0, 0.0, 0.0, 0.0, 0.95]);
    let mut sampler =
        Sampler::with_rng(target, draft, Config::new(2, 8, EOS).unwrap(), rng, 1e-9);
    let r = sampler.step().unwrap();
    assert_eq!(r.proposed, vec![1, 1]);
    assert_eq!(r.accepted, 2);
    assert!(r.rejected_at.is_none());
    assert_eq!(r.bonus_token, Some(2));
    assert_eq!(r.committed, vec![1, 1, 2]);
}
