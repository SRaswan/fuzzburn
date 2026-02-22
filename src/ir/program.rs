//! Root AST nodes and runtime configuration for fuzz programs.

use std::fmt;
use arbitrary::Arbitrary;
use super::ops::{DiffOp, TensorOp};

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
/// FUZZ_NUM_INPUTS=4 FUZZ_MODE=continuous cargo +nightly fuzz run fuzz_autograd
/// ```
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Number of leaf tensors (each requires_grad=true) to build.  The
    /// `AutogradProgram.leaves` field may be shorter or longer; the interpreter
    /// cycles/pads as needed.  Range clamped to [1, 8].
    ///
    /// Env var: `FUZZ_NUM_INPUTS`  (default: 2)
    pub num_inputs: usize,

    /// Whether to panic on the first detected error or log and continue.
    ///
    /// Env var: `FUZZ_MODE`  (`continuous` | `panic`, default: `panic`)
    pub mode: HarnessMode,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig { num_inputs: 2, mode: HarnessMode::PanicOnFirstError }
    }
}

impl FuzzConfig {
    /// Build from environment variables, applying defaults for anything unset.
    pub fn from_env() -> Self {
        let num_inputs = std::env::var("FUZZ_NUM_INPUTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8);

        let mode = match std::env::var("FUZZ_MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "continuous" => HarnessMode::Continuous,
            _ => HarnessMode::PanicOnFirstError,
        };

        FuzzConfig { num_inputs, mode }
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
    /// Render the program as SSA with a concrete `num_inputs` so leaf names
    /// are accurate.  Used in crash reports.
    pub fn ssa(&self, num_inputs: usize) -> String {
        use std::fmt::Write;
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        let mut s = String::new();
        let _ = writeln!(s, "=== AutogradProgram [{}×{}] (num_inputs={}) ===", rows, cols, num_inputs);
        for i in 0..num_inputs {
            let seed_len = self.leaves.get(i).map(|v| v.len()).unwrap_or(0);
            let _ = writeln!(s, "x_{i} = input({}×{}, {seed_len} seed bytes)  [requires_grad]", rows, cols);
        }
        // Initial accumulator is the sum of all leaves.
        let leaf_sum = (0..num_inputs).map(|i| format!("x_{i}")).collect::<Vec<_>>().join(" + ");
        let _ = writeln!(s, "t0 = {leaf_sum}  # all leaves in graph");
        let mut cur = "t0".to_string();
        for (i, op) in self.ops.iter().enumerate() {
            let next = format!("t{}", i + 1);
            // Resolve Leaf refs using the concrete num_inputs.
            let line = match op {
                DiffOp::Add(r) => format!("{next} = {cur} + {}", r.name(num_inputs)),
                DiffOp::Sub(r) => format!("{next} = {cur} - {}", r.name(num_inputs)),
                DiffOp::Mul(r) => format!("{next} = {cur} * {}", r.name(num_inputs)),
                _ => op.ssa_line(&next, &cur),
            };
            let _ = writeln!(s, "{line}");
            cur = next;
        }
        let _ = writeln!(s, "loss = sum({cur})");
        let _ = writeln!(s, "grads = backward(loss)");
        for i in 0..num_inputs {
            let _ = writeln!(s, "assert x_{i}.grad(grads).is_some()");
        }
        s
    }
}

impl fmt::Display for AutogradProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the actual number of leaves the struct holds as a default view.
        write!(f, "{}", self.ssa(self.leaves.len().max(1)))
    }
}