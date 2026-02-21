//! Root AST nodes for the two kinds of fuzz programs.
//!
//! A *program* is the top-level structure libfuzzer generates via `arbitrary`.
//! It owns the tensor shape, the initial data bytes, and the ordered list of
//! ops – think of it as the "module" in an MLIR / LLVM-IR sense.

use arbitrary::Arbitrary;

use super::ops::{DiffOp, TensorOp};

// ─── plain tensor program ─────────────────────────────────────────────────────

/// A fuzzable tensor program that targets the plain (non-differentiable) API.
///
/// ```text
///  t ← from_floats(values, [rows, cols])
///  for op in ops:  t ← op(t)
///  _ ← t.into_data()          // force evaluation
/// ```
#[derive(Arbitrary, Debug)]
pub struct TensorProgram {
    /// Height of the initial 2-D tensor (clamped to [1, 16] at runtime).
    pub rows: u8,
    /// Width of the initial 2-D tensor (clamped to [1, 16] at runtime).
    pub cols: u8,
    /// Raw bytes cycled and normalised to f32 ∈ [-1, 1].
    pub values: Vec<u8>,
    /// The op sequence – the "basic block" of this program.
    pub ops: Vec<TensorOp>,
}

// ─── autograd program ─────────────────────────────────────────────────────────

/// A fuzzable program that targets the autodiff backend.
///
/// ```text
///  a ← from_floats(values_a, [rows, cols]).require_grad()
///  b ← from_floats(values_b, [rows, cols]).require_grad()
///  t ← a
///  for op in ops:  t ← op(t, a, b)       // ops may reference a or b
///  loss ← t.sum()
///  grads ← loss.backward()
///  assert a.grad(&grads).is_some()
///  assert b.grad(&grads).is_some()
/// ```
#[derive(Arbitrary, Debug)]
pub struct AutogradProgram {
    /// Height of both leaf tensors (clamped to [1, 16] at runtime).
    pub rows: u8,
    /// Width of both leaf tensors (clamped to [1, 16] at runtime).
    pub cols: u8,
    /// Initialiser bytes for leaf `a`.
    pub values_a: Vec<u8>,
    /// Initialiser bytes for leaf `b`.
    pub values_b: Vec<u8>,
    /// The differentiable op sequence.
    pub ops: Vec<DiffOp>,
}
