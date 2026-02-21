// IR interpreter:: walks the AST and runs the Burn tensor API.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{activation, Tensor};

use super::ops::{DiffOp, TensorOp, TensorRef};
use super::program::{AutogradProgram, TensorProgram};

type PlainB = NdArray;
type DiffB = Autodiff<NdArray>;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Cycle `raw` bytes and map them to f32 values in [-1, 1].
fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() {
        return vec![0.5_f32; n];
    }
    (0..n)
        .map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0)
        .collect()
}

// ─── plain tensor interpreter ─────────────────────────────────────────────────

/// Execute a [`TensorProgram`] against the plain NdArray backend.
///
/// Panics / aborts are the bugs we are looking for; NaN/Inf outputs are
/// considered valid (floating-point semantics, not bugs).
pub fn run_tensor_program(prog: &TensorProgram) {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);
    let device = NdArrayDevice::default();

    let data = bytes_to_floats(&prog.values, rows * cols);
    let mut t: Tensor<PlainB, 2> =
        Tensor::<PlainB, 1>::from_floats(data.as_slice(), &device).reshape([rows, cols]);

    for op in &prog.ops {
        t = step_tensor(t, op);
    }

    let _ = t.into_data(); // force evaluation
}

/// Single-step evaluation for one [`TensorOp`].
/// Binary ops fold the tensor with a clone of itself (only one value in scope).
fn step_tensor(t: Tensor<PlainB, 2>, op: &TensorOp) -> Tensor<PlainB, 2> {
    match op {
        TensorOp::Add => t.clone() + t.clone(),
        TensorOp::Sub => t.clone() - t.clone(),
        TensorOp::Mul => t.clone() * t.clone(),
        TensorOp::Neg => t.neg(),
        TensorOp::Abs => t.abs(),
        TensorOp::Exp => t.exp(),
        TensorOp::Log => t.log(),
        TensorOp::Sqrt => t.sqrt(),
        TensorOp::Relu => activation::relu(t),
        TensorOp::Sigmoid => activation::sigmoid(t),
        TensorOp::Tanh => activation::tanh(t),
        TensorOp::SumAll => t.sum().unsqueeze::<2>(),
        TensorOp::MeanAll => t.mean().unsqueeze::<2>(),
        TensorOp::Transpose => t.transpose(),
        TensorOp::Clamp => t.clamp(-1e6_f32, 1e6_f32),
    }
}

// ─── autograd interpreter ─────────────────────────────────────────────────────

/// Execute an [`AutogradProgram`] against the Autodiff<NdArray> backend.
///
/// After running all ops, backward is called and we assert that both leaf
/// tensors (a, b) actually received gradients – missing gradients are bugs.
pub fn run_autograd_program(prog: &AutogradProgram) {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);
    let device = NdArrayDevice::default();

    let a: Tensor<DiffB, 2> =
        Tensor::<DiffB, 1>::from_floats(bytes_to_floats(&prog.values_a, rows * cols).as_slice(), &device)
            .reshape([rows, cols])
            .require_grad();

    let b: Tensor<DiffB, 2> =
        Tensor::<DiffB, 1>::from_floats(bytes_to_floats(&prog.values_b, rows * cols).as_slice(), &device)
            .reshape([rows, cols])
            .require_grad();

    // Accumulator starts as a clone of `a`.
    let mut t: Tensor<DiffB, 2> = a.clone();

    for op in &prog.ops {
        t = step_diff(t, &a, &b, op);
    }

    let loss = t.sum();
    let grads = loss.backward();

    // Correctness assertions: every leaf that participated must have a gradient.
    assert!(
        a.grad(&grads).is_some(),
        "grad of leaf `a` missing after backward()"
    );
    assert!(
        b.grad(&grads).is_some(),
        "grad of leaf `b` missing after backward()"
    );
}

/// Resolve a [`TensorRef`] to a concrete tensor at runtime.
///
/// Because `t` might be consumed by the op immediately after, we always clone
/// here – Burn's autodiff tensors are reference-counted, so clones are cheap.
fn resolve(t: &Tensor<DiffB, 2>, a: &Tensor<DiffB, 2>, b: &Tensor<DiffB, 2>, r: &TensorRef) -> Tensor<DiffB, 2> {
    match r {
        TensorRef::A => a.clone(),
        TensorRef::B => b.clone(),
        TensorRef::Current => t.clone(),
    }
}

/// Single-step evaluation for one [`DiffOp`].
///
/// `t` is the current accumulator; `a` and `b` are the named leaf nodes that
/// [`TensorRef`] operands can point to.
fn step_diff(
    t: Tensor<DiffB, 2>,
    a: &Tensor<DiffB, 2>,
    b: &Tensor<DiffB, 2>,
    op: &DiffOp,
) -> Tensor<DiffB, 2> {
    // For binary ops: evaluate LHS (= clone of t) before borrowing t for RHS,
    // so the borrow checker is satisfied and we don't accidentally form cycles.
    match op {
        DiffOp::Add(r) => t.clone() + resolve(&t, a, b, r),
        DiffOp::Sub(r) => t.clone() - resolve(&t, a, b, r),
        DiffOp::Mul(r) => t.clone() * resolve(&t, a, b, r),
        DiffOp::Neg => t.neg(),
        DiffOp::Exp => t.exp(),
        DiffOp::Log => t.log(),
        DiffOp::Sqrt => t.sqrt(),
        DiffOp::Sigmoid => activation::sigmoid(t),
        DiffOp::Tanh => activation::tanh(t),
        DiffOp::SumAll => t.sum().unsqueeze::<2>(),
        DiffOp::Clamp => t.clamp(-1e6_f32, 1e6_f32),
    }
}
