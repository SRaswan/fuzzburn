//! Root AST nodes and runtime configuration for fuzz programs.

use std::fmt;

use arbitrary::Arbitrary;

use super::ops::{DiffOp, TensorOp, TensorRef};

// ─── harness mode ─────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarnessMode {
    PanicOnFirstError,
    Continuous,
}

// ─── fuzz config ──────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Upper bound on the number of distinct leaf tensors that may be used
    pub max_leaves: usize,
    pub mode: HarnessMode,

    /// Shape bounds for rows/cols (keeps fuzzing from OOMing)
    pub min_dim: usize,
    pub max_dim: usize,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig {
            max_leaves: 4,
            mode: HarnessMode::PanicOnFirstError,
            min_dim: 1,
            max_dim: 16,
        }
    }
}

impl FuzzConfig {
    pub fn from_env() -> Self {
        let max_leaves = std::env::var("FUZZ_MAX_LEAVES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 8);

        let mode = match std::env::var("FUZZ_MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "continuous" => HarnessMode::Continuous,
            _ => HarnessMode::PanicOnFirstError,
        };

        let min_dim = std::env::var("FUZZ_MIN_DIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 4096);

        let max_dim = std::env::var("FUZZ_MAX_DIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(min_dim, 4096);

        FuzzConfig {
            max_leaves,
            mode,
            min_dim,
            max_dim,
        }
    }
}

// ─── single-op test case ──────────────────────────────────────────────────────

#[derive(Arbitrary, Debug)]
pub struct SingleOpCase {
    pub rows: u8,
    pub cols: u8,
    pub lhs: Vec<u8>,
    pub rhs: Vec<u8>,
    pub op: TensorOp,
}

impl fmt::Display for SingleOpCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display-only; keep legacy clamp here unless you also want to thread config into Display.
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        writeln!(f, "=== SingleOpCase [{}×{}] ===", rows, cols)?;
        writeln!(f, "a = input({}×{}, {} seed bytes)", rows, cols, self.lhs.len())?;
        writeln!(f, "b = input({}×{}, {} seed bytes)", rows, cols, self.rhs.len())?;
        write!(f, "{}", self.op.ssa_line("result", "a"))
    }
}

// ─── plain tensor program ─────────────────────────────────────────────────────

#[derive(Arbitrary, Debug)]
pub struct TensorProgram {
    pub rows: u8,
    pub cols: u8,
    pub values: Vec<u8>,
    pub ops: Vec<TensorOp>,
}

impl fmt::Display for TensorProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display-only; keep legacy clamp here unless you also want to thread config into Display.
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        writeln!(f, "=== TensorProgram [{}×{}] ===", rows, cols)?;
        writeln!(f, "t0 = input({}×{}, {} seed bytes)", rows, cols, self.values.len())?;
        let mut cur = "t0".to_string();
        for (i, op) in self.ops.iter().enumerate() {
            let next = format!("t{}", i + 1);
            writeln!(f, "{}", op.ssa_line(&next, &cur))?;
            cur = next;
        }
        write!(f, "result = {cur}.into_data()")
    }
}

// ─── autograd program ─────────────────────────────────────────────────────────

/// Fuzzable program targeting the autodiff backend.
/// `leaves` is generated freely by `arbitrary`
/// `TensorRef::Leaf(i)` in `ops` is resolved to `leaves[i % num_inputs]`.
#[derive(Arbitrary, Debug)]
pub struct AutogradProgram {
    pub rows: u8,
    pub cols: u8,
    /// Length may differ from config
    pub leaves: Vec<Vec<u8>>,
    pub ops: Vec<DiffOp>,
}

impl AutogradProgram {
    /// Print that "simulates" dispatch leaf logic.
    pub fn ssa(&self, config: &FuzzConfig) -> String {
        use std::fmt::Write;

        let rows = (self.rows as usize).clamp(config.min_dim, config.max_dim);
        let cols = (self.cols as usize).clamp(config.min_dim, config.max_dim);
        let mut s = String::new();

        let mut num_introduced: usize = 1; // x_0 is always the seed

        let _ = writeln!(
            s,
            "=== AutogradProgram [{}×{}] (max_leaves={}) ===",
            rows, cols, config.max_leaves
        );

        const FALLBACK_SEED_LEN: usize = 8;

        // x_0
        let seed0 = self.leaves.get(0).map(|v| v.len()).unwrap_or(0);
        let seed0_effective = if seed0 == 0 { FALLBACK_SEED_LEN } else { seed0 };
        let seed0_note = if seed0 == 0 { ", fallback" } else { "" };
        let _ = writeln!(
            s,
            "x_0 = input({}×{}, {} seed bytes{})  [requires_grad, seed]",
            rows, cols, seed0_effective, seed0_note
        );

        let _ = writeln!(s, "t0 = x_0  # accumulator seed");

        let mut cur = "t0".to_string();
        for (i, op) in self.ops.iter().enumerate() {
            let next = format!("t{}", i + 1);
            let line = match op {
                DiffOp::Add(r) | DiffOp::Sub(r) | DiffOp::Mul(r) => {
                    let op_sym = match op {
                        DiffOp::Add(_) => "+",
                        DiffOp::Sub(_) => "-",
                        DiffOp::Mul(_) => "*",
                        _ => unreachable!(),
                    };
                    match r {
                        TensorRef::Current => format!("{next} = {cur} {op_sym} {cur}"),
                        TensorRef::Leaf(_) => {
                            if let Some((idx, new_count)) =
                                r.dispatch_leaf_idx(num_introduced, config.max_leaves)
                            {
                                if new_count > num_introduced {
                                    let seed_len =
                                        self.leaves.get(idx).map(|v| v.len()).unwrap_or(0);
                                    let seed_len_effective =
                                        if seed_len == 0 { FALLBACK_SEED_LEN } else { seed_len };
                                    let seed_note =
                                        if seed_len == 0 { ", fallback" } else { "" };

                                    let _ = writeln!(
                                        s,
                                        "x_{idx} = input({}×{}, {} seed bytes{})  \
                                         [requires_grad, introduced here]",
                                        rows, cols, seed_len_effective, seed_note
                                    );
                                    num_introduced = new_count;
                                }
                                format!("{next} = {cur} {op_sym} x_{idx}")
                            } else {
                                format!("{next} = {cur} {op_sym} {cur}")
                            }
                        }
                    }
                }
                _ => op.ssa_line(&next, &cur),
            };

            let _ = writeln!(s, "{line}");
            cur = next;
        }

        let _ = writeln!(s, "grads = backward({cur})");
        for i in 0..num_introduced {
            let _ = writeln!(s, "assert x_{i}.grad(grads).is_some()");
        }
        s
    }
}

impl fmt::Display for AutogradProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cfg = FuzzConfig::default();
        write!(f, "{}", self.ssa(&cfg))
    }
}