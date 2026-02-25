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

    pub safe_math: bool,

    /// 0–100: probability (%) that generator biases to the most recent reg
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
            safe_math: true,
            sink_bias: 20,
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

        let safe_math = std::env::var("FUZZ_SAFE_MATH")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let sink_bias = std::env::var("BIAS")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(20)
            .clamp(0, 100);

        FuzzConfig {
            max_leaves,
            min_ops,
            mode,
            min_dim,
            max_dim,
            safe_math,
            sink_bias,
        }
    }
}

// ─── plain tensor program (SSA) ──────────────────────────────────────────────

#[derive(Debug)]
pub struct TensorProgram {
    pub rows: u8,
    pub cols: u8,
    pub values: Vec<u8>,
    pub ops: Vec<TensorInstr>,
    /// shapes[0] = r0 shape, shapes[i+1] = output shape of ops[i]
    pub shapes: Vec<Shape2>,
}

impl TensorProgram {
    pub fn ssa(&self, config: &FuzzConfig) -> String {
        use std::fmt::Write;

        let mut s = String::new();

        let sh = |reg: usize| -> String {
            self.shapes
                .get(reg)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "[?×?]".into())
        };

        let rows = (self.rows as usize).clamp(config.min_dim, config.max_dim);
        let cols = (self.cols as usize).clamp(config.min_dim, config.max_dim);

        let _ = writeln!(s, "=== TensorProgram [{}×{}] ===", rows, cols);
        let _ = writeln!(
            s,
            "r0 {} = input({}×{}, {} seed bytes)",
            sh(0),
            rows,
            cols,
            self.values.len()
        );

        let mut reg: usize = 1;
        for instr in &self.ops {
            let out = format!("r{reg}");
            let rhs = instr.ssa_line("_").trim_start_matches("_ = ").to_string();
            let _ = writeln!(s, "{out} {} = {rhs}", sh(reg));
            reg += 1;
        }

        let _ = write!(s, "result = r{}.into_data()", reg - 1);
        s
    }
}

impl fmt::Display for TensorProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cfg = FuzzConfig::default();
        write!(f, "{}", self.ssa(&cfg))
    }
}

// ─── autograd program (SSA) ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct AutogradProgram {
    pub rows: u8,
    pub cols: u8,
    pub leaf_seeds: Vec<Vec<u8>>,
    pub ops: Vec<DiffOp>,
    /// shapes[0] = r0 shape, shapes[i+1] = output shape of ops[i]
    pub shapes: Vec<Shape2>,
}

impl AutogradProgram {
    pub fn ssa(&self, config: &FuzzConfig) -> String {
        use std::fmt::Write;

        let mut s = String::new();

        let sh = |reg: usize| -> String {
            self.shapes
                .get(reg)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "[?×?]".into())
        };

        let rows = (self.rows as usize).clamp(config.min_dim, config.max_dim);
        let cols = (self.cols as usize).clamp(config.min_dim, config.max_dim);

        let _ = writeln!(
            s,
            "=== AutogradProgram [{}×{}] (max_leaves={}) ===",
            rows, cols, config.max_leaves
        );

        let seed0_len = self.leaf_seeds.first().map(|v| v.len()).unwrap_or(0);
        let _ = writeln!(
            s,
            "r0 {} = leaf({}×{}, {} seed bytes)  [requires_grad, seed]",
            sh(0),
            rows,
            cols,
            seed0_len
        );

        let mut reg: usize = 1;
        let mut leaf_count: usize = 1;
        let mut leaf_regs: Vec<usize> = vec![0];

        for op in &self.ops {
            let out = format!("r{reg}");
            match op {
                DiffOp::Leaf { seed, rows: lr, cols: lc } => {
                    if leaf_count < config.max_leaves {
                        let pool_idx = if self.leaf_seeds.is_empty() {
                            0
                        } else {
                            *seed as usize % self.leaf_seeds.len()
                        };
                        let seed_len = self.leaf_seeds.get(pool_idx).map(|v| v.len()).unwrap_or(0);

                        let leaf_rows = (*lr as usize).clamp(config.min_dim, config.max_dim);
                        let leaf_cols = (*lc as usize).clamp(config.min_dim, config.max_dim);

                        let _ = writeln!(
                            s,
                            "{out} {} = leaf({leaf_rows}×{leaf_cols}, {seed_len} seed bytes)  \
                             [requires_grad, leaf #{leaf_count}]",
                            sh(reg),
                        );

                        leaf_regs.push(reg);
                        leaf_count += 1;
                    } else {
                        let src = (*seed as usize) % reg;
                        let _ = writeln!(s, "{out} {} = r{src}  # leaf cap reached, alias", sh(reg));
                    }
                }
                _ => {
                    let line = op.ssa_line(&out);
                    let rhs = line.splitn(2, " = ").nth(1).unwrap_or(&line);
                    let _ = writeln!(s, "{out} {} = {rhs}", sh(reg));
                }
            }
            reg += 1;
        }

        let last = reg - 1;
        let _ = writeln!(s, "grads = backward(r{last})");
        for &ri in &leaf_regs {
            let _ = writeln!(s, "grad r{ri} {} = r{ri}.grad(grads)", sh(ri));
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