//! Exact speculative verification over finite-vocabulary tables.
//!
//! Per round the draft distribution proposes up to `draft_window` tokens;
//! each token is accepted against the target distribution using the standard
//! speculative-sampling rule. The marginal output distribution is exactly the
//! target distribution: at the first rejection a token is drawn from the
//! normalized residual `(target - draft)_+ / sum (target - draft)_+`, and
//! every later draft token is discarded.

use crate::checkpoint::{self, SavedState};
use crate::error::{QueryKind, Result, SamplerError};
use crate::model::{validate_distribution, ConditionalTable, Config};
use crate::rng::{Rng, SplitMix64};

/// Why generation stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    /// The round ended but more tokens may be sampled.
    NotFinished,
    /// The end-of-sequence token was committed this round.
    EosCommitted,
    /// The committed sequence reached the configured length bound.
    MaxLengthReached,
}

/// Outcome of the target check at one draft position.
#[derive(Clone, PartialEq, Debug)]
pub struct Acceptance {
    /// Token that was checked.
    pub token: usize,
    /// Prefix of already committed tokens used for the target query.
    pub target_prefix: Vec<usize>,
    /// Prefix used for the (re-queried) draft probability.
    pub draft_prefix: Vec<usize>,
    /// Target mass `P_target(token | prefix)`.
    pub target_mass: f64,
    /// Draft mass `P_draft(token | prefix)`.
    pub draft_mass: f64,
    /// The uniform draw used by the accept/reject test.
    pub draw: f64,
    /// Whether the token was accepted.
    pub accepted: bool,
}

/// Full audit record of one verification round.
#[derive(Clone, PartialEq, Debug)]
pub struct RoundEvent {
    /// Zero-based round index.
    pub round: usize,
    /// Committed sequence immediately before the round.
    pub committed_before: Vec<usize>,
    /// Tokens proposed by the draft (at most `draft_window`).
    pub draft_proposals: Vec<usize>,
    /// Per-position target check results, stopping at the first rejection.
    pub checks: Vec<Acceptance>,
    /// Index into `draft_proposals` of the first rejection, if any.
    pub rejection_index: Option<usize>,
    /// Token drawn from the residual distribution at the rejection.
    pub correction_token: Option<usize>,
    /// The uniform draw used for the correction token.
    pub correction_draw: Option<f64>,
    /// Extra target token sampled when every proposal was accepted.
    pub bonus_token: Option<usize>,
    /// Real prefix used for the bonus target query.
    pub bonus_prefix: Option<Vec<usize>>,
    /// The uniform draw used for the bonus token.
    pub bonus_draw: Option<f64>,
    /// Number of draft tokens accepted this round.
    pub accepted_count: usize,
    /// Committed sequence after the round (the only sequence ever queried).
    pub committed_after: Vec<usize>,
    /// Whether generation is now stopped and why.
    pub stop_reason: StopReason,
}

/// The speculative sampler.
pub struct Sampler<T, D, R = SplitMix64> {
    config: Config,
    target: T,
    draft: D,
    committed: Vec<usize>,
    initial_prefix_len: usize,
    round_index: usize,
    finished: bool,
    rng: R,
}

impl<T, D> Sampler<T, D, SplitMix64>
where
    T: ConditionalTable,
    D: ConditionalTable,
{
    /// Build a sampler seeded by `config.seed`.
    pub fn new(config: Config, target: T, draft: D, initial_prefix: Vec<usize>) -> Result<Self> {
        let rng = SplitMix64::new(config.seed);
        Self::with_rng(config, target, draft, initial_prefix, rng)
    }

    /// Restore a sampler from [`Sampler::save_to_bytes`] output.
    pub fn from_checkpoint(config: Config, target: T, draft: D, bytes: &[u8]) -> Result<Self> {
        let state = checkpoint::decode(bytes)?;
        checkpoint::verify(&state, &config, target.fingerprint(), draft.fingerprint())?;
        Self::from_saved(config, target, draft, state)
    }

    /// Convenience wrapper around [`Sampler::from_checkpoint`] reading a file.
    pub fn restore_from_file(
        config: Config,
        target: T,
        draft: D,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| SamplerError::Io(e.to_string()))?;
        Self::from_checkpoint(config, target, draft, &bytes)
    }
}

impl<T, D, R> Sampler<T, D, R>
where
    T: ConditionalTable,
    D: ConditionalTable,
    R: Rng,
{
    /// Build a sampler with a custom random stream.
    pub fn with_rng(
        config: Config,
        target: T,
        draft: D,
        initial_prefix: Vec<usize>,
        rng: R,
    ) -> Result<Self> {
        for &token in &initial_prefix {
            if token >= config.vocab_size {
                return Err(SamplerError::InvalidPrefixToken(token));
            }
        }
        if initial_prefix.len() > config.max_length {
            return Err(SamplerError::InvalidConfig(format!(
                "initial prefix length {} exceeds max_length {}",
                initial_prefix.len(),
                config.max_length
            )));
        }
        let finished_at_start = initial_prefix.len() >= config.max_length
            || initial_prefix.last() == Some(&config.eos_token);
        Ok(Sampler {
            config,
            target,
            draft,
            initial_prefix_len: initial_prefix.len(),
            committed: initial_prefix,
            round_index: 0,
            finished: finished_at_start,
            rng,
        })
    }

    fn from_saved(config: Config, target: T, draft: D, state: SavedState) -> Result<Self> {
        let prefix = state.committed[..state.initial_prefix_len].to_vec();
        let mut rng = R::from_seed(config.seed);
        rng.set_state(state.rng_state);
        let mut sampler = Self::with_rng(config, target, draft, prefix, rng)?;
        sampler.committed = state.committed;
        sampler.round_index = state.round_index;
        sampler.finished = state.finished;
        Ok(sampler)
    }

    /// Run one full proposal/verification round.
    ///
    /// Returns `None` once generation has stopped. Any error leaves the
    /// committed sequence, round counter and RNG stream untouched, so a later
    /// retry produces exactly the events an uninterrupted run would produce.
    pub fn run_round(&mut self) -> Result<Option<RoundEvent>> {
        if self.finished {
            return Ok(None);
        }
        let event = self.build_round()?;
        self.committed = event.committed_after.clone();
        self.round_index += 1;
        self.finished = event.stop_reason != StopReason::NotFinished;
        Ok(Some(event))
    }

    fn build_round(&mut self) -> Result<RoundEvent> {
        // All draws happen against the live RNG; on any table failure we
        // restore the pre-round state, making the attempt atomic and leaving
        // the future stream identical to an error-free run.
        let saved_rng = self.rng.state();
        match self.build_round_inner() {
            Ok(event) => Ok(event),
            Err(e) => {
                self.rng.set_state(saved_rng);
                Err(e)
            }
        }
    }

    fn build_round_inner(&mut self) -> Result<RoundEvent> {
        let committed_before = self.committed.clone();
        let mut prefix = committed_before.clone();
        let vocab = self.config.vocab_size;

        // ---- draft proposals ------------------------------------------
        let remaining = self.config.max_length - prefix.len();
        let cap = self.config.draft_window.min(remaining);
        let mut proposals = Vec::new();
        for _ in 0..cap {
            let row = fetch(&self.draft, QueryKind::Draft, vocab, &prefix)?;
            let token = sample_weighted(&row, &mut self.rng);
            proposals.push(token);
            prefix.push(token);
            if token == self.config.eos_token {
                break;
            }
        }

        // ---- target verification, stopping at first rejection ---------
        let mut verify_prefix = committed_before.clone();
        let mut checks = Vec::new();
        let mut rejection_index: Option<usize> = None;
        for (index, &token) in proposals.iter().enumerate() {
            let target_row = fetch(&self.target, QueryKind::Target, vocab, &verify_prefix)?;
            let draft_row = fetch(&self.draft, QueryKind::Draft, vocab, &verify_prefix)?;
            let target_mass = target_row[token];
            let draft_mass = draft_row[token];
            let draw = self.rng.next_u01();
            // Accept with probability min(1, target/draft). The multiplicative
            // form avoids a division; zero draft mass always rejects.
            let accepted = draft_mass > 0.0 && draft_mass * draw <= target_mass;
            checks.push(Acceptance {
                token,
                target_prefix: verify_prefix.clone(),
                draft_prefix: verify_prefix.clone(),
                target_mass,
                draft_mass,
                draw,
                accepted,
            });
            if !accepted {
                rejection_index = Some(index);
                break;
            }
            verify_prefix.push(token);
        }

        let mut committed_after = committed_before.clone();
        let accepted_count = rejection_index.unwrap_or(proposals.len());
        committed_after.extend_from_slice(&proposals[..accepted_count]);

        let mut correction_token = None;
        let mut correction_draw = None;
        let mut bonus_token = None;
        let mut bonus_prefix = None;
        let mut bonus_draw = None;
        let mut stop_reason = StopReason::NotFinished;

        match rejection_index {
            Some(index) => {
                let target_row =
                    fetch(&self.target, QueryKind::Target, vocab, &committed_after)?;
                let draft_row =
                    fetch(&self.draft, QueryKind::Draft, vocab, &committed_after)?;
                let residual: Vec<f64> = target_row
                    .iter()
                    .zip(&draft_row)
                    .map(|(q, p)| (q - p).max(0.0))
                    .collect();
                let total: f64 = residual.iter().sum();
                if total <= 0.0 {
                    // Mathematically unreachable: the rejected token had
                    // draft_mass > target_mass, so residual mass is positive.
                    return Err(SamplerError::ZeroResidual {
                        prefix: committed_after.clone(),
                        token: proposals[index],
                    });
                }
                let draw = self.rng.next_u01();
                let token = sample_normalized(&residual, total, draw);
                correction_draw = Some(draw);
                correction_token = Some(token);
                committed_after.push(token);
                if token == self.config.eos_token {
                    stop_reason = StopReason::EosCommitted;
                } else if committed_after.len() >= self.config.max_length {
                    stop_reason = StopReason::MaxLengthReached;
                }
            }
            None => {
                let draft_hit_eos = proposals.last().is_some_and(|&t| t == self.config.eos_token);
                if draft_hit_eos {
                    stop_reason = StopReason::EosCommitted;
                } else if committed_after.len() >= self.config.max_length {
                    stop_reason = StopReason::MaxLengthReached;
                } else {
                    // All proposals accepted: the exact scheme allows one
                    // additional free token drawn directly from the target.
                    let target_row =
                        fetch(&self.target, QueryKind::Target, vocab, &committed_after)?;
                    let draw = self.rng.next_u01();
                    let token = sample_normalized(&target_row, 1.0, draw);
                    bonus_prefix = Some(committed_after.clone());
                    bonus_draw = Some(draw);
                    bonus_token = Some(token);
                    committed_after.push(token);
                    if token == self.config.eos_token {
                        stop_reason = StopReason::EosCommitted;
                    } else if committed_after.len() >= self.config.max_length {
                        stop_reason = StopReason::MaxLengthReached;
                    }
                }
            }
        }

        Ok(RoundEvent {
            round: self.round_index,
            committed_before,
            draft_proposals: proposals,
            checks,
            rejection_index,
            correction_token,
            correction_draw,
            bonus_token,
            bonus_prefix,
            bonus_draw,
            accepted_count,
            committed_after,
            stop_reason,
        })
    }

    /// Current committed sequence (initial prefix plus every accepted token).
    pub fn committed(&self) -> &[usize] {
        &self.committed
    }

    /// The sampler configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Number of completed rounds.
    pub fn round_index(&self) -> usize {
        self.round_index
    }

    /// Whether generation stopped (EOS committed or length bound reached).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Length of the user-supplied initial prefix.
    pub fn initial_prefix_len(&self) -> usize {
        self.initial_prefix_len
    }

    fn saved_state(&self) -> SavedState {
        SavedState {
            config: self.config.clone(),
            target_fp: self.target.fingerprint(),
            draft_fp: self.draft.fingerprint(),
            rng_state: self.rng.state(),
            round_index: self.round_index,
            finished: self.finished,
            initial_prefix_len: self.initial_prefix_len,
            committed: self.committed.clone(),
        }
    }

    /// Serialize a restore point capturing sequence and RNG state.
    pub fn save_to_bytes(&self) -> Vec<u8> {
        checkpoint::encode(&self.saved_state())
    }

    /// Atomically write a restore point (`path.tmp` then rename over `path`).
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("ckpt.tmp");
        std::fs::write(&tmp, self.save_to_bytes()).map_err(|e| SamplerError::Io(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| SamplerError::Io(e.to_string()))
    }

    /// Run a round and, if it committed tokens, atomically save a checkpoint.
    pub fn run_round_checkpointed(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Option<RoundEvent>> {
        let event = self.run_round()?;
        if event.is_some() {
            self.save_to_file(path)?;
        }
        Ok(event)
    }
}

fn fetch<M: ConditionalTable>(
    table: &M,
    kind: QueryKind,
    vocab: usize,
    prefix: &[usize],
) -> Result<Vec<f64>> {
    let row = table.conditional(prefix).map_err(|e| match e {
        SamplerError::MissingDistribution { prefix, .. } => {
            SamplerError::MissingDistribution { kind, prefix }
        }
        SamplerError::InvalidDistribution { prefix, source, .. } => {
            SamplerError::InvalidDistribution {
                kind,
                prefix,
                source,
            }
        }
        other => other,
    })?;
    validate_distribution(&row, vocab).map_err(|source| SamplerError::InvalidDistribution {
        kind,
        prefix: prefix.to_vec(),
        source,
    })?;
    Ok(row)
}

fn sample_weighted(row: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = row.iter().sum();
    sample_normalized(row, total, rng.next_u01())
}

fn sample_normalized(weights: &[f64], total: f64, draw: f64) -> usize {
    let cut = draw * total;
    let mut cum = 0.0;
    let mut last_positive = None;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            last_positive = Some(i);
        }
        cum += w;
        if cut < cum {
            return i;
        }
    }
    last_positive.expect("validated distribution has positive support")
}
