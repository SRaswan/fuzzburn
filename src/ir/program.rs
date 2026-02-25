use std::fmt;

use super::ops::{DiffOp, TensorInstr};
use super::shape::ShapeN;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarnessMode {
    PanicOnFirstError,
    Continuous,
}

#[derive(Debug, Clone)]
pub struct FuzzConfig {
    pub max_leaves: usize,
    pub min_ops: usize,
    pub mode: HarnessMode,

    pub min_dim: usize,
    pub max_dim: usize,

    pub safe_math: bool,
    pub sink_bias: u8,

    pub min_array_elems: usize,
    pub max_array_elems: usize,

    pub tol_rel: f32,
    pub tol_abs: f32,
    pub max_mismatches: usize,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            max_leaves: 4,
            min_ops: 0,
            mode: HarnessMode::PanicOnFirstError,
            min_dim: 1,
            max_dim: 16,
            safe_math: true,
            sink_bias: 20,
            min_array_elems: 1,
            max_array_elems: 256 * 256,
            tol_rel: 1e-4,
            tol_abs: 1e-6,
            max_mismatches: 64,
        }
    }
}

impl FuzzConfig {
    pub fn from_env() -> Self {
        let max_leaves = std::env::var("MAX_LEAVES")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4).clamp(1, 8);

        let min_ops = std::env::var("MIN_OPS")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let mode = match std::env::var("MODE").unwrap_or_default().to_lowercase().as_str() {
            "continuous" => HarnessMode::Continuous,
            _ => HarnessMode::PanicOnFirstError,
        };

        let min_dim = std::env::var("FUZZ_MIN_DIM")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1).clamp(1, 4096);

        let max_dim = std::env::var("FUZZ_MAX_DIM")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16).clamp(min_dim, 4096);

        let safe_math = std::env::var("FUZZ_SAFE_MATH")
            .ok().map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let sink_bias = std::env::var("BIAS")
            .ok().and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(20).clamp(0, 100);

        let min_array_elems = std::env::var("FUZZ_MIN_ARRAY_ELEMS")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1).clamp(1, 1_000_000_000);

        let max_array_elems = std::env::var("FUZZ_MAX_ARRAY_ELEMS")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256*256).clamp(min_array_elems, 1_000_000_000);

        let tol_rel = std::env::var("FUZZ_TOL_REL")
            .ok().and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1e-4).max(0.0);

        let tol_abs = std::env::var("FUZZ_TOL_ABS")
            .ok().and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1e-6).max(0.0);

        let max_mismatches = std::env::var("FUZZ_MAX_MISMATCHES")
            .ok().and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64).clamp(1, 1_000_000);

        Self {
            max_leaves, min_ops, mode,
            min_dim, max_dim,
            safe_math, sink_bias,
            min_array_elems, max_array_elems,
            tol_rel, tol_abs, max_mismatches,
        }
    }
}

#[derive(Debug)]
pub struct TensorProgram {
    pub rows: u8,
    pub cols: u8,
    pub values: Vec<u8>,
    pub ops: Vec<TensorInstr>,
    pub shapes: Vec<ShapeN>, // shapes[0] = r0, shapes[i+1]=out of ops[i]
}

impl TensorProgram {
    pub fn ssa(&self, _config: &FuzzConfig) -> String {
        use std::fmt::Write;
        let mut s = String::new();

        let sh = |reg: usize| -> String {
            self.shapes.get(reg).map(|x| x.to_string()).unwrap_or_else(|| "[?]".into())
        };

        let _ = writeln!(s, "=== TensorProgram ===");
        let _ = writeln!(s, "r0 {} = input({} seed bytes)", sh(0), self.values.len());

        let mut reg = 1usize;
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
        write!(f, "{}", self.ssa(&FuzzConfig::default()))
    }
}

#[derive(Debug)]
pub struct AutogradProgram {
    pub rows: u8,
    pub cols: u8,
    pub leaf_seeds: Vec<Vec<u8>>,
    pub ops: Vec<DiffOp>,
    pub shapes: Vec<ShapeN>, // shapes[0] = r0, shapes[i+1] = out of ops[i]
}

impl AutogradProgram {
    pub fn ssa(&self, config: &FuzzConfig) -> String {
        use std::fmt::Write;
        let mut s = String::new();

        let sh = |reg: usize| -> String {
            self.shapes.get(reg).map(|x| x.to_string()).unwrap_or_else(|| "[?]".into())
        };

        let _ = writeln!(s, "=== AutogradProgram (max_leaves={}) ===", config.max_leaves);
        let seed0_len = self.leaf_seeds.first().map(|v| v.len()).unwrap_or(0);
        let _ = writeln!(s, "r0 {} = leaf({} seed bytes)  [requires_grad, seed]", sh(0), seed0_len);

        let mut reg = 1usize;
        let mut leaf_count = 1usize;
        let mut leaf_regs: Vec<usize> = vec![0];

        for op in &self.ops {
            let out = format!("r{reg}");
            match op {
                DiffOp::Leaf { seed, rank, dims } => {
                    if leaf_count < config.max_leaves {
                        let pool_idx = if self.leaf_seeds.is_empty() { 0 } else { (*seed as usize) % self.leaf_seeds.len() };
                        let seed_len = self.leaf_seeds.get(pool_idx).map(|v| v.len()).unwrap_or(0);

                        // just print shape from shapes[reg] (already computed by generator)
                        let _ = writeln!(
                            s,
                            "{out} {} = leaf(rank={}, dims={:?}, {} seed bytes)  [requires_grad, leaf #{}]",
                            sh(reg),
                            (*rank).clamp(1,4),
                            dims,
                            seed_len,
                            leaf_count
                        );
                        leaf_regs.push(reg);
                        leaf_count += 1;
                    } else {
                        let src = (*seed as usize) % reg;
                        let _ = writeln!(s, "{out} {} = r{src}  # leaf cap reached, alias", sh(reg));
                    }
                }
                DiffOp::Instr(instr) => {
                    let rhs = instr.ssa_line("_").trim_start_matches("_ = ").to_string();
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
        write!(f, "{}", self.ssa(&FuzzConfig::default()))
    }
}