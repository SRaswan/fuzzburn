//! SSA instructions for fuzz IR.
//!
//! The register file is a `Vec<Tensor>`.  Each instruction produces exactly one
//! value and appends it.  Operands are [`Reg(u8)`] references resolved at
//! interpretation time as `raw % num_defined_regs`.

use std::fmt;
use arbitrary::Arbitrary;

// ─── register reference ──────────────────────────────────────────────────────

/// A fuzzer-generated register reference.
/// Resolved at interpretation time: `self.0 as usize % num_regs`.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub struct Reg(pub u8);

impl Reg {
    /// Resolve to a valid register index.  `num_regs` must be > 0.
    #[inline]
    pub fn resolve(&self, num_regs: usize) -> usize {
        (self.0 as usize) % num_regs
    }

    /// Pretty-print using the resolved index.
    pub fn name(&self, num_regs: usize) -> String {
        format!("r{}", self.resolve(num_regs))
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reg({})", self.0)
    }
}

// ─── plain tensor op (SingleOpCase opcode) ───────────────────────────────────

/// Operation enum for isolated single-op testing (`SingleOpCase`).
/// NOT used in SSA programs directly.
#[derive(Arbitrary, Debug, Clone)]
pub enum TensorOp {
    // --- binary ---
    Add,
    Sub,
    Mul,
    // --- unary elementwise ---
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    // --- activations ---
    Relu,
    Sigmoid,
    Tanh,
    // --- reductions ---
    SumAll,
    MeanAll,
    // --- layout ---
    Transpose,
    // --- matmul ---
    Matmul,
    // --- guard ---
    Clamp,
}

impl TensorOp {
    pub fn ssa_line(&self, out: &str, inp: &str) -> String {
        match self {
            TensorOp::Add      => format!("{out} = {inp} + {inp}"),
            TensorOp::Sub      => format!("{out} = {inp} - {inp}"),
            TensorOp::Mul      => format!("{out} = {inp} * {inp}"),
            TensorOp::Neg      => format!("{out} = -{inp}"),
            TensorOp::Abs      => format!("{out} = abs({inp})"),
            TensorOp::Exp      => format!("{out} = exp({inp})"),
            TensorOp::Log      => format!("{out} = log({inp})"),
            TensorOp::Sqrt     => format!("{out} = sqrt({inp})"),
            TensorOp::Relu     => format!("{out} = relu({inp})"),
            TensorOp::Sigmoid  => format!("{out} = sigmoid({inp})"),
            TensorOp::Tanh     => format!("{out} = tanh({inp})"),
            TensorOp::SumAll   => format!("{out} = sum({inp})  # → [1,1]"),
            TensorOp::MeanAll  => format!("{out} = mean({inp})  # → [1,1]"),
            TensorOp::Transpose => format!("{out} = {inp}.T"),
            TensorOp::Matmul   => format!("{out} = {inp} @ {inp}"),
            TensorOp::Clamp    => format!("{out} = clamp({inp}, -1e6, 1e6)"),
        }
    }
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
    // --- unary ---
    Neg(Reg),
    Abs(Reg),
    Exp(Reg),
    Log(Reg),
    Sqrt(Reg),
    // --- activations ---
    Relu(Reg),
    Sigmoid(Reg),
    Tanh(Reg),
    // --- reductions ---
    SumAll(Reg),
    MeanAll(Reg),
    // --- layout ---
    Transpose(Reg),
    // --- matmul ---
    Matmul(Reg, Reg),
    // --- guard ---
    Clamp(Reg),
}

impl TensorInstr {
    /// Pretty-print one SSA line.
    /// `num_regs` = how many registers are defined *before* this instruction.
    pub fn ssa_line(&self, out: &str, num_regs: usize) -> String {
        match self {
            TensorInstr::Add(a, b)    => format!("{out} = {} + {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Sub(a, b)    => format!("{out} = {} - {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Mul(a, b)    => format!("{out} = {} * {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Neg(r)       => format!("{out} = -{}", r.name(num_regs)),
            TensorInstr::Abs(r)       => format!("{out} = abs({})", r.name(num_regs)),
            TensorInstr::Exp(r)       => format!("{out} = exp({})", r.name(num_regs)),
            TensorInstr::Log(r)       => format!("{out} = log({})", r.name(num_regs)),
            TensorInstr::Sqrt(r)      => format!("{out} = sqrt({})", r.name(num_regs)),
            TensorInstr::Relu(r)      => format!("{out} = relu({})", r.name(num_regs)),
            TensorInstr::Sigmoid(r)   => format!("{out} = sigmoid({})", r.name(num_regs)),
            TensorInstr::Tanh(r)      => format!("{out} = tanh({})", r.name(num_regs)),
            TensorInstr::SumAll(r)    => format!("{out} = sum({})  # → [1,1]", r.name(num_regs)),
            TensorInstr::MeanAll(r)   => format!("{out} = mean({})  # → [1,1]", r.name(num_regs)),
            TensorInstr::Transpose(r) => format!("{out} = {}.T", r.name(num_regs)),
            TensorInstr::Matmul(a, b) => format!("{out} = {} @ {}", a.name(num_regs), b.name(num_regs)),
            TensorInstr::Clamp(r)     => format!("{out} = clamp({}, -1e6, 1e6)", r.name(num_regs)),
        }
    }
}

// ─── SSA diff instruction ────────────────────────────────────────────────────

/// SSA instruction for autograd programs.
///
/// `Leaf(u8)` introduces a new `requires_grad` input tensor (the `u8` selects
/// seed data from the seed pool, or aliases a register when `max_leaves` is
/// reached).  All other variants consume register operands and produce a new
/// register value.
#[derive(Arbitrary, Debug, Clone)]
pub enum DiffOp {
    /// Introduce a new leaf input.
    Leaf(u8),
    // --- binary: result = reg ⊕ reg ---
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    // --- unary ---
    Neg(Reg),
    Abs(Reg),
    Exp(Reg),
    Log(Reg),
    Sqrt(Reg),
    // --- activations ---
    Relu(Reg),
    Sigmoid(Reg),
    Tanh(Reg),
    // --- reductions ---
    SumAll(Reg),
    MeanAll(Reg),
    // --- layout ---
    Transpose(Reg),
    // --- matmul ---
    Matmul(Reg, Reg),
    // --- guard ---
    Clamp(Reg),
}

impl DiffOp {
    /// Pretty-print one SSA line.
    /// `num_regs` = how many registers are defined *before* this instruction.
    /// For `Leaf`, a placeholder is printed; callers should override.
    pub fn ssa_line(&self, out: &str, num_regs: usize) -> String {
        match self {
            DiffOp::Leaf(_)       => format!("{out} = leaf()  [requires_grad]"),
            DiffOp::Add(a, b)     => format!("{out} = {} + {}", a.name(num_regs), b.name(num_regs)),
            DiffOp::Sub(a, b)     => format!("{out} = {} - {}", a.name(num_regs), b.name(num_regs)),
            DiffOp::Mul(a, b)     => format!("{out} = {} * {}", a.name(num_regs), b.name(num_regs)),
            DiffOp::Neg(r)        => format!("{out} = -{}", r.name(num_regs)),
            DiffOp::Abs(r)        => format!("{out} = abs({})", r.name(num_regs)),
            DiffOp::Exp(r)        => format!("{out} = exp({})", r.name(num_regs)),
            DiffOp::Log(r)        => format!("{out} = log({})", r.name(num_regs)),
            DiffOp::Sqrt(r)       => format!("{out} = sqrt({})", r.name(num_regs)),
            DiffOp::Relu(r)       => format!("{out} = relu({})", r.name(num_regs)),
            DiffOp::Sigmoid(r)    => format!("{out} = sigmoid({})", r.name(num_regs)),
            DiffOp::Tanh(r)       => format!("{out} = tanh({})", r.name(num_regs)),
            DiffOp::SumAll(r)     => format!("{out} = sum({})  # → [1,1]", r.name(num_regs)),
            DiffOp::MeanAll(r)    => format!("{out} = mean({})  # → [1,1]", r.name(num_regs)),
            DiffOp::Transpose(r)  => format!("{out} = {}.T", r.name(num_regs)),
            DiffOp::Matmul(a, b)  => format!("{out} = {} @ {}", a.name(num_regs), b.name(num_regs)),
            DiffOp::Clamp(r)      => format!("{out} = clamp({}, -1e6, 1e6)", r.name(num_regs)),
        }
    }
}
