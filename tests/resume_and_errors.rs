//! Checkpoint resume equivalence and explicit failure modes.

mod common;

use std::time::{SystemTime, UNIX_EPOCH};

use common::testutil::{RecordingTable, ScriptedRng, ToggleTable};
use exact_draft_sequence_sampler::{
    Checkpoint, Config, MapTable, SampleError, Sampler, Xoshiro256,
};

const EOS: u32 = 0;

fn pref() -> Vec<Vec<u32>> {
    // All prefixes over {1,2} up to length 6, dense enough for every path.
    let mut out = vec![vec![]];
    let mut frontier = vec![vec![]];
    for _ in 0..6 {
        let mut next = Vec::new();
        for p in frontier {
            for t in [1u32, 2] {
                let mut q = p.clone();
                q.push(t);
                out.push(q.clone());
                next.push(q);
            }
        }
        frontier = next;
    }
    out
}
fn rows() -> Vec<(Vec<u32>, Vec<(u32, f64)>)> {
    pref()
        .into_iter()
        .map(|p| (p, vec![(EOS, 0.15), (1, 0.45), (2, 0.40)]))
        .collect()
}
fn drows() -> Vec<(Vec<u32>, Vec<(u32, f64)>)> {
    pref()
        .into_iter()
        .map(|p| (p, vec![(EOS, 0.1), (1, 0.5), (2, 0.4)]))
        .collect()
}

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("edss_{tag}_{}_{}.ckpt", std::process::id(), nanos));
    p
}

#[derive(Clone, PartialEq, Debug)]
struct EventSummary {
    proposed: Vec<u32>,
    accepted: usize,
    rejected_at: Option<usize>,
    repaired: Option<u32>,
    bonus: Option<u32>,
    committed: Vec<u32>,
    sequence: Vec<u32>,
}

fn summarize(r: &exact_draft_sequence_sampler::RoundReport) -> EventSummary {
    EventSummary {
        proposed: r.proposed.clone(),
        accepted: r.accepted,
        rejected_at: r.rejected_at,
        repaired: r.repaired_with,
        bonus: r.bonus_token,
        committed: r.committed.clone(),
        sequence: r.sequence.clone(),
    }
}

#[test]
fn resume_reproduces_uninterrupted_run_exactly() {
    let seed = 20240601u64;
    let path = tmp_path("resume");

    // Uninterrupted reference run, events recorded.
    let mut reference = Sampler::new(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        seed,
    );
    let mut reference_events = Vec::new();
    while !reference.is_finished() {
        reference_events.push(summarize(&reference.step().unwrap()));
    }

    // Run only two rounds, save a checkpoint after each.
    let mut partial = Sampler::new(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        seed,
    );
    partial.step().unwrap();
    partial.step().unwrap();
    partial.save_checkpoint(&path).unwrap();
    let seq_at_resume = partial.sequence().to_vec();

    // Resume from the checkpoint with fresh table instances and continue.
    let mut resumed = Sampler::restore_from_file(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        &path,
    )
    .unwrap();
    assert_eq!(resumed.sequence(), seq_at_resume);
    assert_eq!(resumed.round(), 2);
    let mut resumed_events = Vec::new();
    while !resumed.is_finished() {
        resumed_events.push(summarize(&resumed.step().unwrap()));
    }
    assert_eq!(resumed.sequence(), reference.sequence());
    assert_eq!(&resumed_events, &reference_events[2..]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn reseeding_with_initial_seed_is_not_a_resume() {
    // Demonstrates the requirement: a fresh sampler with the same seed does
    // not reproduce the post-checkpoint stream, whereas true resume does.
    let seed = 777u64;
    let path = tmp_path("reseed");
    let mut run = Sampler::new(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        seed,
    );
    run.step().unwrap();
    run.save_checkpoint(&path).unwrap();
    let true_next = {
        let resumed = Sampler::restore_from_file(
            MapTable::new(rows(), 1e-9).unwrap(),
            MapTable::new(drows(), 1e-9).unwrap(),
            Config::new(2, 6, EOS).unwrap(),
            &path,
        )
        .unwrap();
        let mut r = resumed;
        summarize(&r.step().unwrap())
    };
    let fake_next = {
        // "Resume" by merely resetting to the initial seed: RNG restarts at the
        // beginning instead of the saved stream position.
        let mut fresh = Sampler::new(
            MapTable::new(rows(), 1e-9).unwrap(),
            MapTable::new(drows(), 1e-9).unwrap(),
            Config::new(2, 6, EOS).unwrap(),
            seed,
        );
        summarize(&fresh.step().unwrap())
    };
    assert_ne!(
        true_next, fake_next,
        "re-seeding with the initial seed must not reproduce post-checkpoint events"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn restore_rejects_config_and_table_mismatches() {
    let path = tmp_path("mismatch");
    let mut run = Sampler::new(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        42,
    );
    run.step().unwrap();
    run.save_checkpoint(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let cp = Checkpoint::from_text(&text).unwrap();

    // Different window.
    let mut cp2 = cp.clone();
    cp2.window = 3;
    let err = match Sampler::restore(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        &cp2,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected mismatch error"),
    };
    assert!(matches!(err, SampleError::CheckpointMismatch { .. }));

    // Different max_len.
    let mut cp3 = cp.clone();
    cp3.max_len = 7;
    let err = match Sampler::restore(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        &cp3,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected mismatch error"),
    };
    assert!(matches!(err, SampleError::CheckpointMismatch { .. }));

    // Different target table fingerprint.
    let mut changed = rows();
    changed[0].1[0].1 = 0.20;
    changed[0].1[1].1 = 0.40;
    let err = match Sampler::restore(
        MapTable::new(changed, 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        &cp,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected mismatch error"),
    };
    assert!(matches!(err, SampleError::CheckpointMismatch { .. }));

    // Corrupt checkpoint file.
    std::fs::write(&path, "garbage").unwrap();
    let err = match Sampler::restore_from_file(
        MapTable::new(rows(), 1e-9).unwrap(),
        MapTable::new(drows(), 1e-9).unwrap(),
        Config::new(2, 6, EOS).unwrap(),
        &path,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected corrupt checkpoint error"),
    };
    assert!(matches!(err, SampleError::CheckpointIo { .. } | SampleError::CorruptCheckpoint { .. }));
    std::fs::remove_file(&path).ok();
}

#[test]
fn failed_round_does_not_change_committed_result() {
    // First round succeeds, second round hits an illegal distribution: the
    // sampler must still expose exactly the first-round sequence.
    let flag = std::rc::Rc::new(std::cell::RefCell::new(false));
    let target = ToggleTable::with_flag(
        vec![
            (vec![], vec![(1, 0.6), (2, 0.4)]),
            (vec![1], vec![(1, 0.5), (2, 0.5)]),
            (vec![1, 1], vec![(1, 0.5), (2, 0.5)]),
            (vec![1, 1, 1], vec![(1, 0.5), (2, 0.5)]),
            (vec![1, 1, 1, 1], vec![(1, 0.5), (2, 0.5)]),
            (vec![1, 1, 1, 1, 1], vec![(1, 0.5), (2, 0.5)]),
        ],
        vec![1, 1, 1],
        flag.clone(),
    );
    let draft_rows = vec![
        (vec![], vec![(1, 1.0)]),
        (vec![1], vec![(1, 1.0)]),
        (vec![1, 1], vec![(1, 1.0)]),
        (vec![1, 1, 1], vec![(1, 1.0)]),
        (vec![1, 1, 1, 1], vec![(1, 1.0)]),
        (vec![1, 1, 1, 1, 1], vec![(1, 1.0)]),
    ];
    let draft = RecordingTable::new(draft_rows);
    let rng = ScriptedRng::new(&[0.0; 30]);
    let mut sampler = Sampler::with_rng(
        target,
        draft,
        Config::new(2, 8, EOS).unwrap(),
        rng,
        1e-9,
    );
    let r1 = sampler.step().unwrap();
    // Window fully accepted, so one bonus target token is appended.
    assert_eq!(r1.committed, vec![1, 1, 1]);
    assert_eq!(r1.bonus_token, Some(1));
    *flag.borrow_mut() = true;
    let err = sampler.step().unwrap_err();
    assert!(matches!(err, SampleError::NegativeProbability { .. }));
    assert_eq!(sampler.sequence(), vec![1, 1, 1]);
    assert_eq!(sampler.round(), 1, "failed round must not count");
    *flag.borrow_mut() = false;
    let r3 = sampler.step().unwrap();
    assert_eq!(r3.prefix_before, vec![1, 1, 1]);
}

#[test]
fn missing_prefix_fails_cleanly() {
    let t = MapTable::new(vec![(vec![], vec![(1, 1.0)])], 1e-9).unwrap();
    let d = MapTable::new(
        vec![(vec![], vec![(1, 1.0)]), (vec![1], vec![(1, 1.0)])],
        1e-9,
    )
    .unwrap();
    let mut sampler =
        Sampler::new(t, d, Config::new(2, 4, EOS).unwrap(), 1);
    let err = sampler.step().unwrap_err();
    assert!(matches!(err, SampleError::MissingPrefix { prefix } if prefix == vec![1]));
    assert!(sampler.sequence().is_empty());
}

#[test]
fn table_construction_validates_probabilities() {
    assert!(matches!(
        MapTable::new(vec![(vec![], vec![(1, -0.2), (2, 1.2)])], 1e-9),
        Err(SampleError::NegativeProbability { .. })
    ));
    assert!(matches!(
        MapTable::new(vec![(vec![], vec![(1, f64::NAN)])], 1e-9),
        Err(SampleError::NonFiniteProbability { .. })
    ));
    assert!(matches!(
        MapTable::new(vec![(vec![], vec![(1, 0.3), (2, 0.3)])], 1e-9),
        Err(SampleError::InvalidTotalMass { .. })
    ));
    assert!(matches!(
        MapTable::new(vec![(vec![], vec![(1, 0.0)])], 1e-9),
        Err(SampleError::InvalidTotalMass { .. }) | Err(SampleError::EmptySupport)
    ));
    assert!(matches!(
        Config::new(0, 4, EOS),
        Err(SampleError::InvalidConfig { .. })
    ));
    assert!(matches!(
        Config::new(2, 0, EOS),
        Err(SampleError::InvalidConfig { .. })
    ));
    let _ = Xoshiro256::from_seed(3);
}
