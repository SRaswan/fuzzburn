//! Root AST nodes and runtime configuration for fuzz programs.

use std::fmt;
use arbitrary::Arbitrary;
use super::ops::{DiffOp, TensorOp, TensorRef};

// ─── harness mode ─────────────────────────────────────────────────────────────

/// Controls what happens when an assertion or panic fires in the interpreter.
///
/// | Mode                 | env `FUZZ_MODE`   | Behaviour                              |
/// |----------------------|-------------------|----------------------------------------|
/// | `PanicOnFirstError`  | *(default)*       | re-panic → libFuzzer records crash     |
/// | `Continuous`         | `continuous`      | log to stderr, return → keep running   |
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarnessMode {
    PanicOnFirstError,
    Continuous,
}

// ─── fuzz config ──────────────────────────────────────────────────────────────

/// Runtime configuration read from environment variables by each fuzz target.
///
/// ```text
/// FUZZ_MAX_LEAVES=4 FUZZ_MODE=continuous cargo +nightly fuzz run fuzz_autograd
/// ```
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Upper bound on the number of distinct leaf tensors that may be
    /// introduced during a single program run.  The interpreter discovers the
    /// actual count dynamically via the leaf-stack algorithm; `max_leaves` is
    /// only the ceiling.  Range clamped to [1, 8].
    ///
    /// Env var: `FUZZ_MAX_LEAVES`  (default: 4)
    pub max_leaves: usize,

    /// Whether to panic on the first detected error or log and continue.
    ///
    /// Env var: `FUZZ_MODE`  (`continuous` | `panic`, default: `panic`)
    pub mode: HarnessMode,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig { max_leaves: 4, mode: HarnessMode::PanicOnFirstError }
    }
}

impl FuzzConfig {
    /// Build from environment variables, applying defaults for anything unset.
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

/// A fuzzable program targeting the autodiff backend.
///
/// `leaves` is generated freely by `arbitrary`; the interpreter uses the first
/// `config.num_inputs` entries (padding with empty `Vec`s if needed).
/// `TensorRef::Leaf(i)` in `ops` is resolved to `leaves[i % num_inputs]`.
#[derive(Arbitrary, Debug)]
pub struct AutogradProgram {
    pub rows: u8,
    pub cols: u8,
    /// One initialiser byte-vec per leaf tensor.  Length may differ from
    /// `config.num_inputs`; the interpreter adapts automatically.
    pub leaves: Vec<Vec<u8>>,
    pub ops: Vec<DiffOp>,
}

impl AutogradProgram {
    /// Render the program as SSA by **simulating the leaf-stack algorithm**
    /// with `max_leaves` as the cap.  Leaves are shown as they are introduced
    /// op-by-op, matching exactly what the interpreter will do.  Used in crash
    /// reports.
    pub fn ssa(&self, max_leaves: usize) -> String {
        use std::fmt::Write;
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        let mut s = String::new();

        // Simulate how many leaves get introduced (dry-run of dispatch_leaf_idx).
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
        let _ = writeln!(s, "loss = sum({cur})");
        let _ = writeln!(s, "grads = backward(loss)");
        for i in 0..num_introduced {
            let _ = writeln!(s, "assert x_{i}.grad(grads).is_some()");
        }
        s
    }
}

impl fmt::Display for AutogradProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the default max_leaves for a quick display.
        write!(f, "{}", self.ssa(4))
    }
}