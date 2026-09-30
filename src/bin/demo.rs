//! Small CPU demo: samples from locally defined conditional tables and writes
//! a resume point after every round. Run with `cargo run --release -- --help`.

use std::env;
use std::process::ExitCode;

use exact_draft_sequence_sampler::{
    Config, MapTable, Result, Sampler,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_usize(name: &'static str, raw: Option<String>, default: usize) -> Result<usize> {
    match raw {
        None => Ok(default),
        Some(s) => s.parse::<usize>().map_err(|_| {
            exact_draft_sequence_sampler::SampleError::InvalidConfig {
                field: name,
                reason: format!("not a positive integer: {s}"),
            }
        }),
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut window = 3usize;
    let mut max_len = 20usize;
    let mut seed = 12345u64;
    let mut ckpt = "checkpoint.txt".to_string();
    let mut resume = false;
    let mut i = 0;
    while i < args.len() {
        let takes_value = !matches!(args[i].as_str(), "--resume" | "--help" | "-h");
        match args[i].as_str() {
            "--window" => window = parse_usize("window", args.get(i + 1).cloned(), window)?,
            "--max-len" => max_len = parse_usize("max-len", args.get(i + 1).cloned(), max_len)?,
            "--seed" => seed = parse_usize("seed", args.get(i + 1).cloned(), seed as usize)? as u64,
            "--checkpoint" => ckpt = args.get(i + 1).cloned().unwrap_or(ckpt),
            "--resume" => resume = true,
            "--help" | "-h" => {
                println!(
                    "usage: demo [--window N] [--max-len N] [--seed N] [--checkpoint PATH] [--resume]"
                );
                return Ok(());
            }
            other => {
                return Err(exact_draft_sequence_sampler::SampleError::InvalidConfig {
                    field: "argv",
                    reason: format!("unknown argument: {other}"),
                })
            }
        }
        i += if takes_value { 2 } else { 1 };
    }

    const EOS: u32 = 0;

    // Toy distributions over token ids 1..=3 plus EOS, parameterized by the
    // last token of the real prefix (memoryless given the previous token).
    fn target_entries(last: Option<u32>) -> Vec<(u32, f64)> {
        match last {
            None => vec![(EOS, 0.2), (1, 0.5), (2, 0.2), (3, 0.1)],
            Some(1) => vec![(EOS, 0.3), (1, 0.1), (2, 0.4), (3, 0.2)],
            Some(2) => vec![(EOS, 0.5), (1, 0.2), (2, 0.2), (3, 0.1)],
            _ => vec![(EOS, 0.6), (1, 0.1), (2, 0.1), (3, 0.2)],
        }
    }
    fn draft_entries(last: Option<u32>) -> Vec<(u32, f64)> {
        match last {
            None => vec![(EOS, 0.25), (1, 0.45), (2, 0.2), (3, 0.1)],
            Some(1) => vec![(EOS, 0.25), (1, 0.15), (2, 0.4), (3, 0.2)],
            Some(2) => vec![(EOS, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)],
            _ => vec![(EOS, 0.5), (1, 0.2), (2, 0.1), (3, 0.2)],
        }
    }
    fn dense_rows(
        max_len: usize,
        entries: fn(Option<u32>) -> Vec<(u32, f64)>,
    ) -> Vec<(Vec<u32>, Vec<(u32, f64)>)> {
        let mut rows = vec![(vec![], entries(None))];
        let mut frontier: Vec<Vec<u32>> = vec![vec![]];
        for _ in 0..max_len.saturating_sub(1) {
            let mut next = Vec::new();
            for p in frontier {
                for t in 1u32..=3 {
                    let mut q = p.clone();
                    q.push(t);
                    rows.push((q.clone(), entries(Some(t))));
                    next.push(q);
                }
            }
            frontier = next;
        }
        rows
    }
    let target_rows = dense_rows(max_len, target_entries);
    let draft_rows = dense_rows(max_len, draft_entries);
    let target = MapTable::new(target_rows, 1e-9)?;
    let draft = MapTable::new(draft_rows, 1e-9)?;
    let config = Config::new(window, max_len, EOS)?;

    let mut sampler = if resume {
        println!("resuming from {ckpt}");
        Sampler::restore_from_file(target, draft, config, &ckpt)?
    } else {
        Sampler::new(target, draft, config, seed)
    };

    while !sampler.is_finished() {
        let report = sampler.step()?;
        sampler.save_checkpoint(&ckpt)?;
        println!(
            "round {}: proposed={:?} accepted={} rejected_at={:?} repaired={:?} bonus={:?} committed={:?} seq={:?} finished={}",
            report.round,
            report.proposed,
            report.accepted,
            report.rejected_at,
            report.repaired_with,
            report.bonus_token,
            report.committed,
            report.sequence,
            report.finished
        );
    }
    println!("final sequence: {:?}", sampler.sequence());
    Ok(())
}
