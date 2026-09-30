#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use exact_draft_sequence_sampler::{
    ConditionalTable, Distribution, Result, SampleError, UniformSource,
};

/// Scripted stream of uniforms: lets tests pin every accept/reject/draw event.
#[derive(Default)]
pub struct ScriptedRng {
    pub draws: RefCell<Vec<f64>>,
    pub consumed: RefCell<Vec<f64>>,
}

impl ScriptedRng {
    pub fn new(draws: &[f64]) -> Self {
        Self {
            draws: RefCell::new(draws.to_vec()),
            consumed: RefCell::new(Vec::new()),
        }
    }
}

impl UniformSource for ScriptedRng {
    fn uniform(&mut self) -> f64 {
        let mut d = self.draws.borrow_mut();
        let u = if d.is_empty() {
            panic!("scripted rng exhausted")
        } else {
            d.remove(0)
        };
        self.consumed.borrow_mut().push(u);
        u
    }
}

/// A table built from validated rows that also records every queried prefix,
/// so tests can prove queries use the real prefix and never touch discarded
/// draft positions.
pub struct RecordingTable {
    rows: HashMap<Vec<u32>, Vec<(u32, f64)>>,
    pub queries: Rc<RefCell<Vec<(Which, Vec<u32>)>>>,
    which: Which,
    fp: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Target,
    Draft,
}

impl RecordingTable {
    pub fn new(rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>) -> Self {
        Self::with_log(rows, Rc::new(RefCell::new(Vec::new())), Which::Target)
    }

    pub fn pair(
        target_rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>,
        draft_rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>,
    ) -> (Self, Self, Rc<RefCell<Vec<(Which, Vec<u32>)>>>) {
        let log = Rc::new(RefCell::new(Vec::new()));
        let t = Self::with_log(target_rows, log.clone(), Which::Target);
        let d = Self::with_log(draft_rows, log.clone(), Which::Draft);
        (t, d, log)
    }

    fn with_log(
        rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>,
        log: Rc<RefCell<Vec<(Which, Vec<u32>)>>>,
        which: Which,
    ) -> Self {
        let mut map = HashMap::new();
        let mut fp = 0u64;
        for (prefix, entries) in rows {
            fp ^= simple_hash(&prefix, &entries);
            map.insert(prefix, entries);
        }
        Self {
            rows: map,
            queries: log,
            which,
            fp,
        }
    }
}

impl ConditionalTable for RecordingTable {
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution> {
        self.queries.borrow_mut().push((self.which, prefix.to_vec()));
        let entries = self
            .rows
            .get(prefix)
            .ok_or_else(|| SampleError::MissingPrefix {
                prefix: prefix.to_vec(),
            })?
            .clone();
        Distribution::new(entries, 1e-9)
    }

    fn fingerprint(&self) -> u64 {
        self.fp
    }
}

/// Handle used by tests to flip a row of [`ToggleTable`] after construction.
pub trait BadFlag {
    fn set_bad(&self, bad: bool);
}

impl BadFlag for Rc<RefCell<bool>> {
    fn set_bad(&self, bad: bool) {
        *self.borrow_mut() = bad;
    }
}

/// Table whose row for one specific prefix can be flipped between a valid row
/// and an illegal one (used to show a failed round changes nothing).
pub struct ToggleTable {
    valid: HashMap<Vec<u32>, Vec<(u32, f64)>>,
    bad: Rc<RefCell<bool>>,
    bad_prefix: Vec<u32>,
    fp: u64,
}

impl ToggleTable {
    pub fn new(
        rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>,
        bad_prefix: Vec<u32>,
    ) -> Self {
        Self::with_flag(rows, bad_prefix, Rc::new(RefCell::new(false)))
    }

    pub fn with_flag(
        rows: Vec<(Vec<u32>, Vec<(u32, f64)>)>,
        bad_prefix: Vec<u32>,
        bad: Rc<RefCell<bool>>,
    ) -> Self {
        let mut map = HashMap::new();
        let mut fp = 0u64;
        for (prefix, entries) in rows {
            fp ^= simple_hash(&prefix, &entries);
            map.insert(prefix, entries);
        }
        Self {
            valid: map,
            bad,
            bad_prefix,
            fp,
        }
    }

    pub fn set_bad(&self, bad: bool) {
        *self.bad.borrow_mut() = bad;
    }
}

impl ConditionalTable for ToggleTable {
    fn distribution(&self, prefix: &[u32]) -> Result<Distribution> {
        if *self.bad.borrow() && prefix == self.bad_prefix {
            return Distribution::new(vec![(1, -0.5), (2, 1.5)], 1e-9);
        }
        let entries = self
            .valid
            .get(prefix)
            .ok_or_else(|| SampleError::MissingPrefix {
                prefix: prefix.to_vec(),
            })?
            .clone();
        Distribution::new(entries, 1e-9)
    }

    fn fingerprint(&self) -> u64 {
        self.fp
    }
}

fn simple_hash(prefix: &[u32], entries: &[(u32, f64)]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &t in prefix {
        for byte in (t as u64).to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    for (t, p) in entries {
        for byte in (*t as u64).to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        for byte in p.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}
