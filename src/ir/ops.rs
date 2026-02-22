//! Instructions for fuzz IR.
//! TensorRef is the "operand" type for binary ops

use std::fmt;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug, Clone)]
pub enum TensorRef {
    /// Leaf tensor. byte drives TensorRef::dispatch_leaf_idx
    Leaf(u8),
    /// Current accumulated value (output of previous op)
    Current,
}

impl TensorRef {
    pub fn dispatch_leaf_idx(&self, num_used: usize, max_leaves: usize) -> Option<(usize, usize)> {
        match self {
            TensorRef::Current => None,
            TensorRef::Leaf(raw) => {
                let raw = *raw as usize;
                let num_available = max_leaves.saturating_sub(num_used);

                Some(if num_used == 0 {
                    // Stack is empty: always introduce x_0.
                    (0, 1)
                } else if num_available == 0 {
                    // Stack full: must reuse
                    (raw % num_used, num_used)
                } else {
                    // Reuse var probability = num_used / (num_used + num_avail)
                    let threshold = (num_used * 256) / (num_used + num_available);
                    if raw < threshold {
                        (raw % num_used, num_used)
                    } else {
                        // Make new leaf
                        (num_used, num_used + 1)
                    }
                })
            }
        }
    }

    pub fn name(&self, num_used: usize, max_leaves: usize) -> String {
        match self.dispatch_leaf_idx(num_used, max_leaves) {
            None => "t".to_string(),
            Some((idx, _)) => format!("x_{idx}"),
        }
    }
}

impl fmt::Display for TensorRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorRef::Leaf(i) => write!(f, "x_{}", i),
            TensorRef::Current => write!(f, "t"),
        }
    }
}

// plain tensor ops
// ─────────────────────────────────────────────────────────

/// TODO: more layout ops?  indexing/slicing ops?  (currently we just have Transpose)
/// TODO: type/shape inference?
/// TODO: differentiable backend ops should be compatible
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

// SSA print
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
            TensorOp::Clamp    => format!("{out} = clamp({inp}, -1e6, 1e6)"),
        }
    }
}

// differentiable ops 
// ───────────────────────────────────────────────────────

/// differentiable subset of ops
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

impl DiffOp {
    pub fn ssa_line(&self, out: &str, inp: &str) -> String {
        match self {
            DiffOp::Add(r)  => format!("{out} = {inp} + {r}"),
            DiffOp::Sub(r)  => format!("{out} = {inp} - {r}"),
            DiffOp::Mul(r)  => format!("{out} = {inp} * {r}"),
            DiffOp::Neg     => format!("{out} = -{inp}"),
            DiffOp::Exp     => format!("{out} = exp({inp})"),
            DiffOp::Log     => format!("{out} = log({inp})"),
            DiffOp::Sqrt    => format!("{out} = sqrt({inp})"),
            DiffOp::Sigmoid => format!("{out} = sigmoid({inp})"),
            DiffOp::Tanh    => format!("{out} = tanh({inp})"),
            DiffOp::SumAll  => format!("{out} = sum({inp})  # → [1,1]"),
            DiffOp::Clamp   => format!("{out} = clamp({inp}, -1e6, 1e6)"),
        }
    }
}
