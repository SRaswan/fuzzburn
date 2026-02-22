//! Lightweight 2-D shape tracking for the register file.

use super::super::ops::{Reg, TensorInstr, DiffOp};

/// Lightweight 2-D shape tracked alongside every register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape2(pub usize, pub usize);

impl Shape2 {
    /// Can two 2-D shapes be element-wise combined under NumPy broadcasting?
    #[inline]
    pub fn broadcast_compatible(self, other: Shape2) -> bool {
        (self.0 == other.0 || self.0 == 1 || other.0 == 1)
            && (self.1 == other.1 || self.1 == 1 || other.1 == 1)
    }

    /// Result shape after broadcasting two compatible shapes.
    #[inline]
    pub fn broadcast_result(self, other: Shape2) -> Shape2 {
        Shape2(self.0.max(other.0), self.1.max(other.1))
    }

    /// Can `self` be left-multiplied by `other`?  i.e. `self @ other`
    /// requires self.cols == other.rows.
    #[inline]
    pub fn matmul_compatible(self, other: Shape2) -> bool {
        self.1 == other.0
    }

    /// Given a unary/binary TensorInstr, compute the output shape.
    pub fn after_tensor_instr(shapes: &[Shape2], instr: &TensorInstr) -> Shape2 {
        let n = shapes.len();
        match instr {
            TensorInstr::Add(a, b)
            | TensorInstr::Sub(a, b)
            | TensorInstr::Mul(a, b) => {
                let sa = shapes[a.resolve(n)];
                let sb = shapes[resolve_broadcast_compatible(shapes, a.resolve(n), b)];
                sa.broadcast_result(sb)
            }
            TensorInstr::Matmul(a, b) => {
                let sa = shapes[a.resolve(n)];
                match resolve_matmul_compatible(shapes, a.resolve(n), b) {
                    Some(bi) => Shape2(sa.0, shapes[bi].1),
                    None => sa, // demoted to passthrough
                }
            }
            TensorInstr::Neg(r)
            | TensorInstr::Abs(r)
            | TensorInstr::Exp(r)
            | TensorInstr::Log(r)
            | TensorInstr::Sqrt(r)
            | TensorInstr::Relu(r)
            | TensorInstr::Sigmoid(r)
            | TensorInstr::Tanh(r)
            | TensorInstr::Clamp(r) => shapes[r.resolve(n)],
            TensorInstr::SumAll(_) | TensorInstr::MeanAll(_) => Shape2(1, 1),
            TensorInstr::Transpose(r) => {
                let Shape2(r_, c_) = shapes[r.resolve(n)];
                Shape2(c_, r_)
            }
        }
    }

    /// Given a DiffOp, compute the output shape.  Returns `None` for `Leaf`
    /// (handled by the main loop).
    pub fn after_diff_op(shapes: &[Shape2], op: &DiffOp) -> Option<Shape2> {
        match op {
            DiffOp::Leaf { .. } => None,
            DiffOp::Instr(instr) => Some(Shape2::after_tensor_instr(shapes, instr)),
        }
    }
}

/// For element-wise binary operations (Add/Sub/Mul), resolve operand `b` to a
/// register whose shape is **broadcast-compatible** with operand `a`.
/// Falls back to `a` itself when nothing compatible exists.
pub(crate) fn resolve_broadcast_compatible(shapes: &[Shape2], a_idx: usize, b_raw: &Reg) -> usize {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.broadcast_compatible(shapes[b_idx]) {
        return b_idx;
    }
    for i in (0..n).rev() {
        if sa.broadcast_compatible(shapes[i]) {
            return i;
        }
    }
    a_idx // ultimate fallback: a ⊕ a
}

/// For matmul, resolve operand `b` to a register where
/// `shapes[a_idx].cols == shapes[b_idx].rows`.
/// Returns `None` when no compatible register exists.
pub(crate) fn resolve_matmul_compatible(shapes: &[Shape2], a_idx: usize, b_raw: &Reg) -> Option<usize> {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.matmul_compatible(shapes[b_idx]) {
        return Some(b_idx);
    }
    for i in (0..n).rev() {
        if sa.matmul_compatible(shapes[i]) {
            return Some(i);
        }
    }
    None
}
