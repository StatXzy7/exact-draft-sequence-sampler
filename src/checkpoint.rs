//! Self-contained binary save/restore points.
//!
//! Layout (all integers little-endian):
//! `MAGIC(4) | VERSION(u16) | payload | FNV1a64(payload)`
//!
//! The payload records the *full* RNG state, not just the initial seed, plus
//! every configuration field, fingerprints of both tables and the complete
//! committed sequence. Restoring therefore resumes exactly; merely re-seeding
//! a fresh sampler can never reproduce an interrupted run.

use crate::error::{Result, SamplerError};
use crate::model::{fnv1a_bytes, fnv1a_init, Config};

const MAGIC: &[u8; 4] = b"EDSS";
const VERSION: u16 = 1;

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct SavedState {
    pub config: Config,
    pub target_fp: u64,
    pub draft_fp: u64,
    pub rng_state: u64,
    pub round_index: usize,
    pub finished: bool,
    pub initial_prefix_len: usize,
    pub committed: Vec<usize>,
}

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer {
            buf: Vec::with_capacity(128),
        }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(SamplerError::CorruptCheckpoint("unexpected end of buffer".into()));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut a = [0u8; 2];
        a.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(a))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }
    fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?)
            .map_err(|_| SamplerError::CorruptCheckpoint("usize out of range".into()))
    }
}

pub(crate) fn encode(state: &SavedState) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(MAGIC);
    w.u16(VERSION);
    let payload_start = w.buf.len();

    w.usize(state.config.vocab_size);
    w.usize(state.config.draft_window);
    w.usize(state.config.max_length);
    w.usize(state.config.eos_token);
    w.u64(state.config.seed);
    w.u64(state.target_fp);
    w.u64(state.draft_fp);
    w.u64(state.rng_state);
    w.usize(state.round_index);
    w.u8(state.finished as u8);
    w.usize(state.initial_prefix_len);
    w.usize(state.committed.len());
    for &token in &state.committed {
        w.usize(token);
    }

    let mut hash = fnv1a_init();
    fnv1a_bytes(&mut hash, &w.buf[payload_start..]);
    w.u64(hash);
    w.buf
}

pub(crate) fn decode(bytes: &[u8]) -> Result<SavedState> {
    if bytes.len() < 4 + 2 + 8 {
        return Err(SamplerError::CorruptCheckpoint("buffer too short".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(SamplerError::CorruptCheckpoint("bad magic".into()));
    }
    let mut r = Reader::new(bytes);
    r.take(4)?;
    let version = r.u16()?;
    if version != VERSION {
        return Err(SamplerError::CorruptCheckpoint(format!(
            "unsupported version {version}"
        )));
    }
    let payload_start = 6;
    let stored_checksum = {
        if bytes.len() < 8 {
            return Err(SamplerError::CorruptCheckpoint("missing checksum".into()));
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&bytes[bytes.len() - 8..]);
        u64::from_le_bytes(a)
    };
    let mut hash = fnv1a_init();
    fnv1a_bytes(&mut hash, &bytes[payload_start..bytes.len() - 8]);
    if hash != stored_checksum {
        return Err(SamplerError::CorruptCheckpoint("checksum mismatch".into()));
    }

    let vocab_size = r.usize()?;
    let draft_window = r.usize()?;
    let max_length = r.usize()?;
    let eos_token = r.usize()?;
    let seed = r.u64()?;
    let target_fp = r.u64()?;
    let draft_fp = r.u64()?;
    let rng_state = r.u64()?;
    let round_index = r.usize()?;
    let finished = match r.u8()? {
        0 => false,
        1 => true,
        other => {
            return Err(SamplerError::CorruptCheckpoint(format!(
                "invalid finished flag {other}"
            )))
        }
    };
    let initial_prefix_len = r.usize()?;
    let committed_len = r.usize()?;
    let mut committed = Vec::with_capacity(committed_len.min(1 << 28));
    for _ in 0..committed_len {
        let token = r.usize()?;
        if token >= vocab_size {
            return Err(SamplerError::CorruptCheckpoint(
                "saved token outside vocabulary".into(),
            ));
        }
        committed.push(token);
    }
    if r.pos + 8 != bytes.len() {
        return Err(SamplerError::CorruptCheckpoint("trailing bytes".into()));
    }
    let config = Config::new(vocab_size, draft_window, max_length, eos_token, seed)
        .map_err(|e| SamplerError::CorruptCheckpoint(e.to_string()))?;
    if initial_prefix_len > committed.len() || committed.len() > max_length {
        return Err(SamplerError::CorruptCheckpoint(
            "inconsistent saved lengths".into(),
        ));
    }
    Ok(SavedState {
        config,
        target_fp,
        draft_fp,
        rng_state,
        round_index,
        finished,
        initial_prefix_len,
        committed,
    })
}

pub(crate) fn verify(
    state: &SavedState,
    config: &Config,
    target_fp: u64,
    draft_fp: u64,
) -> Result<()> {
    let mismatch = |field: &str| SamplerError::ConfigMismatch {
        field: field.to_string(),
    };
    if state.config.vocab_size != config.vocab_size {
        return Err(mismatch("vocab_size"));
    }
    if state.config.draft_window != config.draft_window {
        return Err(mismatch("draft_window"));
    }
    if state.config.max_length != config.max_length {
        return Err(mismatch("max_length"));
    }
    if state.config.eos_token != config.eos_token {
        return Err(mismatch("eos_token"));
    }
    if state.config.seed != config.seed {
        return Err(mismatch("seed"));
    }
    if state.target_fp != target_fp || state.draft_fp != draft_fp {
        return Err(SamplerError::ModelMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SavedState {
        SavedState {
            config: Config::new(3, 2, 8, 0, 99).unwrap(),
            target_fp: 1,
            draft_fp: 2,
            rng_state: 0xdead_beef,
            round_index: 4,
            finished: false,
            initial_prefix_len: 1,
            committed: vec![2, 1, 0],
        }
    }

    #[test]
    fn roundtrip() {
        let s = sample();
        let bytes = encode(&s);
        let back = decode(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn tampering_is_detected() {
        let s = sample();
        let mut bytes = encode(&s);
        let n = bytes.len();
        bytes[n - 10] ^= 0xff;
        assert!(decode(&bytes).is_err());
        bytes[n - 1] = bytes[n - 1].wrapping_add(1);
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn verify_detects_field_and_model_mismatch() {
        let s = sample();
        let ok = Config::new(3, 2, 8, 0, 99).unwrap();
        assert!(verify(&s, &ok, 1, 2).is_ok());
        assert!(matches!(
            verify(&s, &Config::new(3, 3, 8, 0, 99).unwrap(), 1, 2),
            Err(SamplerError::ConfigMismatch { .. })
        ));
        assert!(matches!(
            verify(&s, &ok, 9, 2),
            Err(SamplerError::ModelMismatch)
        ));
    }
}
