//! Root AST nodes and runtime configuration for fuzz programs.

use std::fmt;
use super::ops::{DiffOp, TensorInstr};
use super::shape::Shape2;

// ─── harness mode ─────────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarnessMode {
    PanicOnFirstError,
    Continuous,
}

// ─── fuzz config ──────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct FuzzConfig {
    /// Upper bound on the number of distinct leaf tensors that may be introduced
    pub max_leaves: usize,
    /// Minimum number of ops a program must have; smaller inputs are skipped.
    pub min_ops: usize,
    pub mode: HarnessMode,

    pub min_dim: usize,
    pub max_dim: usize,

    /// 0–100: probability (%) that `pick_reg` returns the last-written register
    /// (the "sink") instead of picking uniformly.  Higher = deeper chains.
    pub sink_bias: u8,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        FuzzConfig {
            max_leaves: 4,
            min_ops: 0,
            mode: HarnessMode::PanicOnFirstError,
            min_dim: 1,
            max_dim: 16,
            sink_bias: 60,
        }
    }
}

impl FuzzConfig {
    pub fn from_env() -> Self {
        let max_leaves = std::env::var("MAX_LEAVES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 8);

        let min_ops = std::env::var("MIN_OPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let mode = match std::env::var("MODE")
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

        let sink_bias = std::env::var("BIAS")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(25)
            .clamp(0, 100);

        FuzzConfig {
            max_leaves,
            min_ops,
            mode,
            min_dim,
            max_dim,
            sink_bias,
        }
    }
}
// ─── plain tensor program (SSA) ──────────────────────────────────────────────

/// SSA tensor program.  `r0` is seeded from `values`; every [`TensorInstr`]
/// appends a new register to the file.
#[derive(Debug)]
pub struct TensorProgram {
    pub rows: u8,
    pub cols: u8,
    pub values: Vec<u8>,
    pub ops: Vec<TensorInstr>,
    /// Pre-computed shape of every register.  `shapes[0]` is r0, `shapes[i+1]`
    /// is the output of `ops[i]`.  Populated by the generator.
    pub shapes: Vec<Shape2>,
}

impl fmt::Display for TensorProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        writeln!(f, "=== TensorProgram [{}×{}] ===", rows, cols)?;

        let sh = |reg: usize| -> String {
            self.shapes
                .get(reg)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "[?×?]".into())
        };

        writeln!(f, "r0 {} = input({}×{}, {} seed bytes)", sh(0), rows, cols, self.values.len())?;
        let mut num_regs: usize = 1;
        for instr in &self.ops {
            let out = format!("r{}", num_regs);
            writeln!(f, "{} {} = {}", out, sh(num_regs), instr.ssa_line("_").trim_start_matches("_ = "))?;
            num_regs += 1;
        }
        write!(f, "result = r{}.into_data()", num_regs - 1)
    }
}

// ─── autograd program (SSA) ──────────────────────────────────────────────────

/// SSA autograd program.
///
/// `r0` is always the seed leaf (`requires_grad`).  [`DiffOp::Leaf`]
/// instructions introduce additional leaves (up to `max_leaves`).  All other
/// [`DiffOp`] variants reference registers by [`Reg`] and push new values.
///
/// The register file is a flat `Vec<Tensor>` — leaves and intermediates share
/// the same index space, so the fuzzer can freely compose any DAG.
#[derive(Debug)]
pub struct AutogradProgram {
    pub rows: u8,
    pub cols: u8,
    /// Pool of seed byte-vectors for leaf tensors.
    pub leaf_seeds: Vec<Vec<u8>>,
    pub ops: Vec<DiffOp>,
    /// Pre-computed shape of every register.  `shapes[0]` is r0, `shapes[i+1]`
    /// is the output of `ops[i]`.  Populated by the generator; the interpreter
    /// trusts these instead of recomputing.
    pub shapes: Vec<Shape2>,
}

impl AutogradProgram {
    /// Pretty-print the program in SSA form using pre-computed shapes.
    pub fn ssa(&self) -> String {
        use std::fmt::Write;

        let rows = (self.rows as usize).clamp(1, 16);
        let cols = (self.cols as usize).clamp(1, 16);
        let mut s = String::new();
        let _ = writeln!(s, "=== AutogradProgram [{}×{}] ===", rows, cols);

        let sh = |reg: usize| -> String {
            self.shapes.get(reg).map(|s| s.to_string()).unwrap_or_else(|| "[?×?]".into())
        };

        let _ = writeln!(s, "r0 {} = leaf({}×{})  [seed]", sh(0), rows, cols);

        let mut num_regs: usize = 1;
        let mut leaf_regs: Vec<usize> = vec![0];

        for op in &self.ops {
            let _ = write!(s, "r{} {} = ", num_regs, sh(num_regs));
            match op {
                DiffOp::Leaf { rows: lr, cols: lc, .. } => {
                    let _ = writeln!(s, "leaf({}×{})  [leaf #{}]",
                        (*lr as usize).clamp(1, 16), (*lc as usize).clamp(1, 16), leaf_regs.len());
                    leaf_regs.push(num_regs);
                }
                DiffOp::Instr(instr) => {
                    let line = instr.ssa_line("_");
                    let _ = writeln!(s, "{}", line.trim_start_matches("_ = "));
                }
            }
            num_regs += 1;
        }

        let last = num_regs - 1;
        let _ = writeln!(s, "grads = backward(r{last})");
        for &ri in &leaf_regs {
            let _ = writeln!(s, "grad r{ri} {}", sh(ri));
        }
        s
    }
}

impl fmt::Display for AutogradProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ssa())
    }
}