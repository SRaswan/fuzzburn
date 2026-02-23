//! SSA instructions for fuzz IR.
//!
//! The register file is a `Vec<Tensor>`.  Each instruction produces exactly one
//! value and appends it.  Operands are [`Reg(u8)`] indices guaranteed valid by
//! the state-machine generator.

use std::fmt;
use arbitrary::Arbitrary;

// ─── register reference ──────────────────────────────────────────────────────

/// A register index into the SSA file.  The generator guarantees the index is
/// in-bounds, so no modulo resolution is needed.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub struct Reg(pub u8);

impl Reg {
    /// Pretty-print as `rN`.
    pub fn name(self) -> String {
        format!("r{}", self.0)
    }
}

impl From<Reg> for usize {
    #[inline]
    fn from(r: Reg) -> usize { r.0 as usize }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

// ─── operation category (for the state-machine generator) ────────────────────

/// High-level category of operation the generator should produce next.
///
/// Deriving `Arbitrary` lets the fuzzer engine decide the distribution instead
/// of a hardcoded `int_in_range` lookup table.  The generator's state machine
/// still fills in shape-legal operands for whichever category is drawn.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub enum OpCategory {
    Unary,
    Binary,
    Matmul,
    Transpose,
    DimReduce,
    FullReduce,
    Concat,
    Repeat,
    Powf,
    Slice,
}

/// Which element-wise unary / activation to apply.
///
/// Same idea — `Arbitrary` picks; the generator just wraps the chosen variant
/// around a register reference that is already known to be valid.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub enum UnaryKind {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Cos,
    Sin,
    Relu,
    Sigmoid,
    Tanh,
    Clamp,
}

/// Which element-wise binary op to apply.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

// ─── SSA tensor instruction ──────────────────────────────────────────────────

/// SSA instruction for plain tensor programs.
/// Each instruction consumes register operand(s) and produces one new register.
#[derive(Arbitrary, Debug, Clone)]
pub enum TensorInstr {
    // --- binary: result = reg ⊕ reg ---
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),
    // --- unary ---
    Neg(Reg),
    Abs(Reg),
    Exp(Reg),
    Log(Reg),
    Sqrt(Reg),
    Cos(Reg),
    Sin(Reg),
    // --- activations ---
    Relu(Reg),
    Sigmoid(Reg),
    Tanh(Reg),
    // --- reductions ---
    SumAll(Reg),
    MeanAll(Reg),
    // --- dimensional reductions ---
    SumDim(Reg, u8),
    MeanDim(Reg, u8),
    ArgMax(Reg, u8),
    // --- layout ---
    Transpose(Reg),
    Concat(Reg, Reg, u8),
    Repeat(Reg, u8, u8),
    Slice(Reg, u8, u8),
    // --- matmul ---
    Matmul(Reg, Reg),
    // --- parametric ---
    Powf(Reg, u8),
    // --- guard ---
    Clamp(Reg),
}

impl TensorInstr {
    /// Pretty-print one SSA line.
    pub fn ssa_line(&self, out: &str) -> String {
        match self {
            TensorInstr::Add(a, b)    => format!("{out} = {} + {}", a.name(), b.name()),
            TensorInstr::Sub(a, b)    => format!("{out} = {} - {}", a.name(), b.name()),
            TensorInstr::Mul(a, b)    => format!("{out} = {} * {}", a.name(), b.name()),
            TensorInstr::Div(a, b)    => format!("{out} = {} / {}", a.name(), b.name()),
            TensorInstr::Neg(r)       => format!("{out} = -{}", r.name()),
            TensorInstr::Abs(r)       => format!("{out} = abs({})", r.name()),
            TensorInstr::Exp(r)       => format!("{out} = exp({})", r.name()),
            TensorInstr::Log(r)       => format!("{out} = log({})", r.name()),
            TensorInstr::Sqrt(r)      => format!("{out} = sqrt({})", r.name()),
            TensorInstr::Cos(r)       => format!("{out} = cos({})", r.name()),
            TensorInstr::Sin(r)       => format!("{out} = sin({})", r.name()),
            TensorInstr::Relu(r)      => format!("{out} = relu({})", r.name()),
            TensorInstr::Sigmoid(r)   => format!("{out} = sigmoid({})", r.name()),
            TensorInstr::Tanh(r)      => format!("{out} = tanh({})", r.name()),
            TensorInstr::SumAll(r)    => format!("{out} = sum({})", r.name()),
            TensorInstr::MeanAll(r)   => format!("{out} = mean({})", r.name()),
            TensorInstr::SumDim(r, d)  => {
                let dim = *d as usize % 2;
                format!("{out} = sum({}, dim={dim})", r.name())
            }
            TensorInstr::MeanDim(r, d) => {
                let dim = *d as usize % 2;
                format!("{out} = mean({}, dim={dim})", r.name())
            }
            TensorInstr::ArgMax(r, d)  => {
                let dim = *d as usize % 2;
                format!("{out} = argmax({}, dim={dim})", r.name())
            }
            TensorInstr::Transpose(r) => format!("{out} = {}.T", r.name()),
            TensorInstr::Concat(a, b, d) => {
                let dim = *d as usize % 2;
                format!("{out} = cat([{}, {}], dim={dim})", a.name(), b.name())
            }
            TensorInstr::Repeat(r, d, c) => {
                let dim = *d as usize % 2;
                let count = (*c as usize).clamp(1, 4);
                format!("{out} = {}.repeat(dim={dim}, ×{count})", r.name())
            }
            TensorInstr::Slice(r, d, len) => {
                let dim = *d as usize % 2;
                format!("{out} = {}.slice(dim={dim}, 0..{len})", r.name())
            }
            TensorInstr::Matmul(a, b) => format!("{out} = {} @ {}", a.name(), b.name()),
            TensorInstr::Powf(r, c)   => {
                const EXPONENTS: [f32; 5] = [0.5, 2.0, 3.0, -1.0, 1.5];
                let exp = EXPONENTS[(*c as usize) % EXPONENTS.len()];
                format!("{out} = {}.powf({exp})", r.name())
            }
            TensorInstr::Clamp(r)     => format!("{out} = clamp({}, -1e6, 1e6)", r.name()),
        }
    }
}

// ─── SSA diff instruction ────────────────────────────────────────────────────

/// SSA instruction for autograd programs.
///
/// `Leaf { seed, rows, cols }` introduces a new `requires_grad` input tensor
/// with its own shape.  The `seed` byte selects data from the seed pool (or
/// aliases a register when `max_leaves` is reached).  `Instr` wraps a
/// [`TensorInstr`] — all the same ops, no duplication.
#[derive(Arbitrary, Debug, Clone)]
pub enum DiffOp {
    /// Introduce a new leaf input with its own shape.
    Leaf {
        seed: u8,
        rows: u8,
        cols: u8,
    },
    /// Any tensor instruction (shared with plain-tensor programs).
    Instr(TensorInstr),
}

impl DiffOp {
    /// Pretty-print one SSA line.
    /// For `Leaf`, a placeholder is printed; callers should override.
    pub fn ssa_line(&self, out: &str) -> String {
        match self {
            DiffOp::Leaf { rows, cols, .. } => {
                let r = (*rows as usize).clamp(1, 16);
                let c = (*cols as usize).clamp(1, 16);
                format!("{out} = leaf({r}×{c})  [requires_grad]")
            }
            DiffOp::Instr(i) => i.ssa_line(out),
        }
    }
}
