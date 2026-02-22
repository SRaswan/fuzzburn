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
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig { max_leaves: 4, mode: HarnessMode::PanicOnFirstError }
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

        FuzzConfig { max_leaves, mode }
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
    // print that "simulates" dispatch leaf logic
    pub fn ssa(&self, max_leaves: usize) -> String {
        use std::fmt::Write;
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        let mut s = String::new();

        let mut num_introduced: usize = 1; // x_0 is always the seed

        let _ = writeln!(s, "=== AutogradProgram [{}×{}] (max_leaves={}) ===", rows, cols, max_leaves);
        let seed0 = self.leaves.get(0).map(|v| v.len()).unwrap_or(0);
        let _ = writeln!(s, "x_0 = input({}×{}, {seed0} seed bytes)  [requires_grad, seed]", rows, cols);
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
                                r.dispatch_leaf_idx(num_introduced, max_leaves)
                            {
                                if new_count > num_introduced {
                                    let seed_len =
                                        self.leaves.get(idx).map(|v| v.len()).unwrap_or(0);
                                    let _ = writeln!(
                                        s,
                                        "x_{idx} = input({}×{}, {seed_len} seed bytes)  \
                                         [requires_grad, introduced here]",
                                        rows, cols
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
        write!(f, "{}", self.ssa(4))
    }
}