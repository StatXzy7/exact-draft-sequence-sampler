//! Speculative draft-and-verify sampler with the exact target output law.
//!
//! One `step` proceeds as follows:
//! 1. propose up to `window` draft tokens, querying the draft table only at
//!    the real committed prefix plus already accepted draft tokens;
//! 2. verify each proposed token against the target distribution at the same
//!    prefix with probability `min(1, p/q)`;
//! 3. the first rejection discards that token and every later draft token; a
//!    single repair token is drawn from the normalized rejection residual
//!    `(p - q)_+ / sum` (the disjoint-support case is exactly a direct target
//!    draw);
//! 4. if every proposal is accepted, one extra target token is sampled;
//! 5. generation stops immediately once the EOS token is committed, and the
//!    output never exceeds `max_len`.

use std::fs;
use std::path::Path;

use crate::dist::{acceptance_prob, residual_distribution};
use crate::error::{Result, SampleError};
use crate::rng::{UniformSource, Xoshiro256};
use crate::table::ConditionalTable;

/// User configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Maximum number of draft tokens proposed in a single round (`>= 1`).
    pub window: usize,
    /// Hard upper bound on the length of the committed sequence.
    pub max_len: usize,
    /// End-of-sequence token; the run stops as soon as it is committed.
    pub eos: u32,
}

impl Config {
    /// Construct and validate a configuration.
    pub fn new(window: usize, max_len: usize, eos: u32) -> Result<Self> {
        if window == 0 {
            return Err(SampleError::InvalidConfig {
                field: "window",
                reason: "must be at least 1".into(),
            });
        }
        if max_len == 0 {
            return Err(SampleError::InvalidConfig {
                field: "max_len",
                reason: "must be at least 1".into(),
            });
        }
        Ok(Self { window, max_len, eos })
    }

    /// Reconstruct a config from checkpoint fields; structural corruption is
    /// reported as a corrupt checkpoint rather than a fresh-config error.
    fn from_checkpoint(window: usize, max_len: usize, eos: u32) -> Result<Self> {
        if window == 0 || max_len == 0 {
            return Err(SampleError::CorruptCheckpoint {
                detail: format!("window={window} max_len={max_len}"),
            });
        }
        Ok(Self { window, max_len, eos })
    }
}

/// Everything observable about one verification round.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundReport {
    /// Zero-based round counter (incremented on every `step`, including a
    /// finished no-op).
    pub round: usize,
    /// Tokens the draft proposed this round (before any discarding).
    pub proposed: Vec<u32>,
    /// Number of proposals accepted in order (a contiguous prefix).
    pub accepted: usize,
    /// Index in `proposed` of the first rejection, if any.
    pub rejected_at: Option<usize>,
    /// Token drawn from the rejection residual, if any.
    pub repaired_with: Option<u32>,
    /// Extra target token sampled when the whole window was accepted.
    pub bonus_token: Option<u32>,
    /// The committed prefix *before* the round.
    pub prefix_before: Vec<u32>,
    /// Tokens actually committed this round, in order.
    pub committed: Vec<u32>,
    /// Committed sequence after the round.
    pub sequence: Vec<u32>,
    /// True iff the run has stopped (EOS committed or `max_len` reached).
    pub finished: bool,
}

impl RoundReport {
    /// Convenience: prefix against which the rejected position was verified.
    pub fn rejected_prefix(&self) -> Option<Vec<u32>> {
        let pos = self.rejected_at?;
        let mut p = self.prefix_before.clone();
        p.extend_from_slice(&self.proposed[..pos]);
        Some(p)
    }
}

/// The draft-and-verify sampler.
pub struct Sampler<T, D, R = Xoshiro256>
where
    T: ConditionalTable,
    D: ConditionalTable,
    R: UniformSource,
{
    target: T,
    draft: D,
    config: Config,
    target_fp: u64,
    draft_fp: u64,
    tol: f64,
    rng: R,
    sequence: Vec<u32>,
    round: usize,
}

const DEFAULT_TOL: f64 = 1e-9;

impl<T, D> Sampler<T, D, Xoshiro256>
where
    T: ConditionalTable,
    D: ConditionalTable,
{
    /// Start a fresh run seeded by `seed`.
    pub fn new(target: T, draft: D, config: Config, seed: u64) -> Self {
        Self::with_rng(
            target,
            draft,
            config,
            Xoshiro256::from_seed(seed),
            DEFAULT_TOL,
        )
    }
}

impl<T, D, R> Sampler<T, D, R>
where
    T: ConditionalTable,
    D: ConditionalTable,
    R: UniformSource,
{
    /// Construct with a custom uniform stream (mainly for deterministic tests)
    /// and a custom mass tolerance.
    pub fn with_rng(target: T, draft: D, config: Config, rng: R, tol: f64) -> Self {
        let target_fp = target.fingerprint();
        let draft_fp = draft.fingerprint();
        Self {
            target,
            draft,
            config,
            target_fp,
            draft_fp,
            tol,
            rng,
            sequence: Vec::new(),
            round: 0,
        }
    }

    /// Committed sequence so far.
    pub fn sequence(&self) -> &[u32] {
        &self.sequence
    }

    /// Current configuration.
    pub fn config(&self) -> Config {
        self.config
    }

    /// Number of rounds already executed.
    pub fn round(&self) -> usize {
        self.round
    }

    /// True once EOS has been committed or `max_len` has been reached.
    pub fn is_finished(&self) -> bool {
        self.sequence.len() >= self.config.max_len
            || self
                .sequence
                .last()
                .map(|t| *t == self.config.eos)
                .unwrap_or(false)
    }
}

struct RoundOutcome {
    proposed: Vec<u32>,
    accepted: usize,
    rejected_at: Option<usize>,
    repaired_with: Option<u32>,
    bonus_token: Option<u32>,
    committed: Vec<u32>,
}

impl<T, D, R> Sampler<T, D, R>
where
    T: ConditionalTable,
    D: ConditionalTable,
    R: UniformSource,
{
    /// Run one draft-propose / target-verify round and commit the result.
    ///
    /// Any error aborts the round *before* the committed sequence is touched:
    /// tables queried mid-round, illegal mass and missing prefixes all leave
    /// the sampler exactly as it was before the call (except that, as for any
    /// failed operation, no checkpoint is written).
    pub fn step(&mut self) -> Result<RoundReport> {
        let prefix_before = self.sequence.clone();

        if self.is_finished() {
            return Ok(RoundReport {
                round: self.round,
                proposed: Vec::new(),
                accepted: 0,
                rejected_at: None,
                repaired_with: None,
                bonus_token: None,
                prefix_before,
                committed: Vec::new(),
                sequence: self.sequence.clone(),
                finished: true,
            });
        }

        let outcome = self.run_round()?;
        self.sequence.extend_from_slice(&outcome.committed);
        self.round += 1;
        Ok(RoundReport {
            round: self.round,
            proposed: outcome.proposed,
            accepted: outcome.accepted,
            rejected_at: outcome.rejected_at,
            repaired_with: outcome.repaired_with,
            bonus_token: outcome.bonus_token,
            prefix_before,
            committed: outcome.committed,
            sequence: self.sequence.clone(),
            finished: self.is_finished(),
        })
    }

    fn run_round(&mut self) -> Result<RoundOutcome> {
        let mut proposed: Vec<u32> = Vec::new();
        let mut accepted = 0usize;
        let mut rejected_at: Option<usize> = None;
        let mut repaired_with: Option<u32> = None;
        let mut bonus_token: Option<u32> = None;
        let mut committed: Vec<u32> = Vec::new();

        let room = self.config.max_len - self.sequence.len();
        let budget = self.config.window.min(room);

        // Propose and verify one position at a time. The draft for position k
        // is queried at the real committed prefix plus the contiguous accepted
        // prefix; after the first rejection no further draft is ever proposed
        // or queried.
        let mut prefix = self.sequence.clone();
        for k in 0..budget {
            let q = self.draft.distribution(&prefix)?;
            let token = q.draw(&mut self.rng)?;
            proposed.push(token);

            let p = self.target.distribution(&prefix)?;
            let ratio = acceptance_prob(p.mass(token), q.mass(token));
            let accept_u = self.rng.uniform();
            if accept_u < ratio {
                accepted += 1;
                committed.push(token);
                prefix.push(token);
                if token == self.config.eos {
                    // EOS committed: stop immediately, discard later proposals.
                    break;
                }
            } else {
                rejected_at = Some(k);
                let residual = residual_distribution(&p, &q, self.tol)?;
                let repair = match residual {
                    Some(dist) => dist.draw(&mut self.rng)?,
                    // Identical/covered distributions give zero residual mass,
                    // but then rejection probability is zero; reaching here is
                    // an explicit failure rather than a division by zero.
                    None => return Err(SampleError::ZeroResidualMass),
                };
                repaired_with = Some(repair);
                committed.push(repair);
                break;
            }
        }

        // Whole window accepted without EOS: one extra direct target token,
        // never beyond the hard length cap.
        if rejected_at.is_none() && committed.len() < room {
            let last_is_eos = committed.last().map(|t| *t == self.config.eos).unwrap_or(false);
            if !last_is_eos {
                let p = self.target.distribution(&prefix)?;
                let extra = p.draw(&mut self.rng)?;
                bonus_token = Some(extra);
                committed.push(extra);
            }
        }

        Ok(RoundOutcome {
            proposed,
            accepted,
            rejected_at,
            repaired_with,
            bonus_token,
            committed,
        })
    }
}

const CHECKPOINT_VERSION: u32 = 1;

/// Snapshot written after each committed round. A resume reconstructs not only
/// the sequence but also the exact internal random stream position; reseeding
/// with the initial seed cannot produce an equivalent state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub version: u32,
    pub window: usize,
    pub max_len: usize,
    pub eos: u32,
    pub target_fp: u64,
    pub draft_fp: u64,
    pub round: usize,
    pub sequence: Vec<u32>,
    pub rng_state: [u64; 4],
}

impl Checkpoint {
    fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("exact-draft-sequence-sampler v{}\n", self.version));
        s.push_str(&format!("window={}\n", self.window));
        s.push_str(&format!("max_len={}\n", self.max_len));
        s.push_str(&format!("eos={}\n", self.eos));
        s.push_str(&format!("target_fp={}\n", self.target_fp));
        s.push_str(&format!("draft_fp={}\n", self.draft_fp));
        s.push_str(&format!("round={}\n", self.round));
        s.push_str(&format!(
            "sequence={}\n",
            self.sequence
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
        s.push_str(&format!(
            "rng_state={},{},{},{}\n",
            self.rng_state[0],
            self.rng_state[1],
            self.rng_state[2],
            self.rng_state[3]
        ));
        s
    }

    pub fn from_text(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| SampleError::CorruptCheckpoint {
                detail: "empty file".into(),
            })?;
        let version = header
            .strip_prefix("exact-draft-sequence-sampler v")
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or_else(|| SampleError::CorruptCheckpoint {
                detail: format!("bad header: {header}"),
            })?;
        let get = |key: &str| -> Result<String> {
            let needle = format!("{key}=");
            lines
                .clone()
                .find_map(|line| line.strip_prefix(&needle).map(str::to_string))
                .ok_or_else(|| SampleError::CorruptCheckpoint {
                    detail: format!("missing key {key}"),
                })
        };
        let parse = |key: &str, raw: String| -> Result<u64> {
            raw.parse::<u64>().map_err(|_| SampleError::CorruptCheckpoint {
                detail: format!("bad integer for {key}: {raw}"),
            })
        };
        let window = parse("window", get("window")?)? as usize;
        let max_len = parse("max_len", get("max_len")?)? as usize;
        let eos = parse("eos", get("eos")?)? as u32;
        let target_fp = parse("target_fp", get("target_fp")?)?;
        let draft_fp = parse("draft_fp", get("draft_fp")?)?;
        let round = parse("round", get("round")?)? as usize;
        let seq_raw = get("sequence")?;
        let sequence = if seq_raw.is_empty() {
            Vec::new()
        } else {
            seq_raw
                .split(',')
                .map(|t| {
                    t.parse::<u32>().map_err(|_| SampleError::CorruptCheckpoint {
                        detail: format!("bad token in sequence: {t}"),
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        let state_raw = get("rng_state")?;
        let words: Vec<u64> = state_raw
            .split(',')
            .map(|w| {
                w.parse::<u64>().map_err(|_| SampleError::CorruptCheckpoint {
                    detail: format!("bad rng word: {w}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if words.len() != 4 {
            return Err(SampleError::CorruptCheckpoint {
                detail: format!("rng_state must have 4 words, got {}", words.len()),
            });
        }
        Ok(Checkpoint {
            version,
            window,
            max_len,
            eos,
            target_fp,
            draft_fp,
            round,
            sequence,
            rng_state: [words[0], words[1], words[2], words[3]],
        })
    }
}

impl<T, D, R> Sampler<T, D, R>
where
    T: ConditionalTable,
    D: ConditionalTable,
    R: RngCheckpoint,
{
    /// Snapshot the current run.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            version: CHECKPOINT_VERSION,
            window: self.config.window,
            max_len: self.config.max_len,
            eos: self.config.eos,
            target_fp: self.target_fp,
            draft_fp: self.draft_fp,
            round: self.round,
            sequence: self.sequence.clone(),
            rng_state: self.rng.state_words(),
        }
    }

    /// Write the snapshot to `path`.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = self.checkpoint().to_text();
        fs::write(path, text).map_err(|e| SampleError::CheckpointIo {
            path: path.display().to_string(),
            detail: e.to_string(),
        })
    }
}

/// Random streams that can expose/reconstruct their full state for resume.
pub trait RngCheckpoint: UniformSource {
    fn state_words(&self) -> [u64; 4];
    fn from_state_words(words: [u64; 4]) -> Self;
}

impl RngCheckpoint for Xoshiro256 {
    fn state_words(&self) -> [u64; 4] {
        self.state()
    }
    fn from_state_words(words: [u64; 4]) -> Self {
        Xoshiro256::from_state(words)
    }
}

impl<T, D> Sampler<T, D, Xoshiro256>
where
    T: ConditionalTable,
    D: ConditionalTable,
{
    /// Restore a run against an explicitly supplied configuration. Fails
    /// without side effects when the checkpoint was made for another config or
    /// other tables (fingerprints must match).
    pub fn restore(
        target: T,
        draft: D,
        config: Config,
        cp: &Checkpoint,
    ) -> Result<Self> {
        if cp.version != CHECKPOINT_VERSION {
            return Err(SampleError::CheckpointMismatch {
                detail: format!("unsupported version {}", cp.version),
            });
        }
        let stored = Config::from_checkpoint(cp.window, cp.max_len, cp.eos)?;
        let mismatch = |field: &str| SampleError::CheckpointMismatch {
            detail: format!("{field} differs from checkpoint"),
        };
        if config != stored {
            return Err(mismatch("configuration"));
        }
        if target.fingerprint() != cp.target_fp {
            return Err(mismatch("target table"));
        }
        if draft.fingerprint() != cp.draft_fp {
            return Err(mismatch("draft table"));
        }
        if cp.sequence.len() > cp.max_len {
            return Err(SampleError::CorruptCheckpoint {
                detail: "sequence longer than max_len".into(),
            });
        }
        let mut sampler = Self::with_rng(
            target,
            draft,
            config,
            RngCheckpoint::from_state_words(cp.rng_state),
            DEFAULT_TOL,
        );
        sampler.sequence = cp.sequence.clone();
        sampler.round = cp.round;
        Ok(sampler)
    }

    /// Read, validate and restore from a checkpoint file.
    pub fn restore_from_file(
        target: T,
        draft: D,
        config: Config,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|e| SampleError::CheckpointIo {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        let cp = Checkpoint::from_text(&text)?;
        Self::restore(target, draft, config, &cp)
    }
}
