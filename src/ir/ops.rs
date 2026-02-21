//! The instruction set for the fuzz IR.
//!
//! Think of this as the opcode table: every enum variant is one node in the
//! tensor-program AST.  Ops are split into two sets:
//!
//!  - [`TensorOp`]  – all ops supported by a plain (non-differentiable) backend.
//!  - [`DiffOp`]    – the differentiable subset, safe to use with autodiff.
//!
//! [`TensorRef`] is the "operand" type for binary ops in the autograd world:
//! it lets the fuzzer choose whether the RHS of an Add/Sub/Mul should be the
//! first leaf, the second leaf, or the current accumulator – giving the
//! interpreter a real (directed acyclic) compute graph rather than a linear
//! chain.

use arbitrary::Arbitrary;

// ─── operand reference ────────────────────────────────────────────────────────

/// Names a tensor in scope during autograd program execution.
#[derive(Arbitrary, Debug, Clone)]
pub enum TensorRef {
    /// The first leaf input (requires_grad = true).
    A,
    /// The second leaf input (requires_grad = true).
    B,
    /// The current accumulated value (the output of the previous op).
    Current,
}

// ─── plain tensor ops ─────────────────────────────────────────────────────────

/// Every tensor op we can fuzz against a non-differentiable backend.
/// Binary ops always use `self ⊕ self`; there is only one tensor in scope.
#[derive(Arbitrary, Debug, Clone)]
pub enum TensorOp {
    // --- binary (self ⊕ self) ---
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
    // --- reductions (output becomes [1, 1] for all following ops) ---
    SumAll,
    MeanAll,
    // --- layout ---
    Transpose,
    // --- guard against runaway values ---
    Clamp,
}

// ─── differentiable ops ───────────────────────────────────────────────────────

/// The differentiable subset of ops.  Binary ops carry a [`TensorRef`] so the
/// interpreter can build a real DAG instead of a straight chain.
#[derive(Arbitrary, Debug, Clone)]
pub enum DiffOp {
    // --- binary: t ⊕ ref ---
    Add(TensorRef),
    Sub(TensorRef),
    Mul(TensorRef),
    // --- unary ---
    Neg,
    Exp,
    Log,
    Sqrt,
    // --- activations ---
    Sigmoid,
    Tanh,
    // --- reduction ---
    SumAll,
    // --- guard ---
    Clamp,
}
