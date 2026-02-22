// IR interpreter – walks the AST and runs the Burn tensor API.

use std::panic;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{activation, Tensor};

use super::ops::{DiffOp, TensorOp, TensorRef};
use super::program::{AutogradProgram, FuzzConfig, HarnessMode, TensorProgram};

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

/// Decide what to do after catching a panic, based on `HarnessMode`.
///
/// - `PanicOnFirstError` → resume the unwind (libFuzzer records the crash)
/// - `Continuous`        → log to stderr and return (fuzzer keeps running)
fn handle_crash(e: Box<dyn std::any::Any + Send>, display: &str, target: &str, mode: HarnessMode) {
    eprintln!("\n=== CRASH DETECTED ({target}) ===");
    eprintln!("{display}");
    eprintln!("========================================\n");
    match mode {
        HarnessMode::PanicOnFirstError => panic::resume_unwind(e),
        HarnessMode::Continuous => { /* log only – let the fuzzer continue */ }
    }
}

// ─── plain tensor interpreter ─────────────────────────────────────────────────

/// Execute a [`TensorProgram`] against the plain NdArray backend.
///
/// `mode` controls whether a detected crash re-panics (libFuzzer records it)
/// or is logged to stderr and execution continues.
pub fn run_tensor_program(prog: &TensorProgram, mode: HarnessMode) {
    let display = prog.to_string();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_tensor_program_inner(prog);
    }));
    if let Err(e) = result {
        handle_crash(e, &display, "fuzz_tensor_ops", mode);
    }
}

fn run_tensor_program_inner(prog: &TensorProgram) {
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
/// Leaves are discovered dynamically: `x_0` always seeds the accumulator, and
/// each binary op's [`TensorRef::Leaf`] byte drives the stack algorithm
/// ([`TensorRef::dispatch_leaf_idx`]), introducing new leaves or reusing
/// existing ones up to `config.max_leaves`.
///
/// `config.mode` controls crash behaviour (re-panic vs. continuous logging).
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) {
    let display = prog.ssa(config.max_leaves);
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_autograd_program_inner(prog, config);
    }));
    if let Err(e) = result {
        handle_crash(e, &display, "fuzz_autograd", config.mode);
    }
}

fn run_autograd_program_inner(prog: &AutogradProgram, config: &FuzzConfig) {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);
    let device = NdArrayDevice::default();

    // x_0 is unconditionally the accumulator seed.  It is always reachable in
    // the compute graph (loss = t.sum() flows back through every op to x_0).
    let leaf_0 = make_leaf(
        prog.leaves.get(0).map(Vec::as_slice).unwrap_or(&[]),
        rows, cols, &device,
    );
    let mut leaf_stack: Vec<Tensor<DiffB, 2>> = vec![leaf_0.clone()];
    let mut t: Tensor<DiffB, 2> = leaf_0;

    // Run the op sequence.  Binary ops may grow leaf_stack via the stack
    // dispatch algorithm, building a real DAG (not just a chain).
    for op in &prog.ops {
        t = step_diff(t, op, &mut leaf_stack, &prog.leaves, config.max_leaves, rows, cols, &device);
    }

    let loss = t.sum();
    let grads = loss.backward();

    // Every leaf in leaf_stack was introduced because it appeared in an op
    // that contributed to the computation graph, so a missing gradient here
    // is a genuine Burn bug, not a scaffolding artifact.
    for (i, leaf) in leaf_stack.iter().enumerate() {
        if leaf.grad(&grads).is_none() {
            panic!("grad of leaf x_{i} missing after backward()");
        }
    }
}

/// Build one `requires_grad` leaf tensor from raw seed bytes.
fn make_leaf(raw: &[u8], rows: usize, cols: usize, device: &NdArrayDevice) -> Tensor<DiffB, 2> {
    Tensor::<DiffB, 1>::from_floats(
        bytes_to_floats(raw, rows * cols).as_slice(),
        device,
    )
    .reshape([rows, cols])
    .require_grad()
}

/// Resolve a [`TensorRef`] using the leaf-stack dispatch algorithm.
///
/// - `TensorRef::Current` → clone the accumulator `t`.
/// - `TensorRef::Leaf(raw)` → reuse or introduce a leaf via
///   [`TensorRef::dispatch_leaf_idx`], growing `leaf_stack` when a new leaf
///   is introduced.
fn resolve_ref(
    r: &TensorRef,
    t: &Tensor<DiffB, 2>,
    leaf_stack: &mut Vec<Tensor<DiffB, 2>>,
    leaf_data: &[Vec<u8>],
    max_leaves: usize,
    rows: usize,
    cols: usize,
    device: &NdArrayDevice,
) -> Tensor<DiffB, 2> {
    match r.dispatch_leaf_idx(leaf_stack.len(), max_leaves) {
        None => t.clone(), // TensorRef::Current
        Some((idx, new_count)) => {
            if new_count > leaf_stack.len() {
                // Introduce a new leaf at position `idx` (= old stack length).
                let raw = leaf_data.get(idx).map(Vec::as_slice).unwrap_or(&[]);
                leaf_stack.push(make_leaf(raw, rows, cols, device));
            }
            leaf_stack[idx].clone()
        }
    }
}

/// Single-step evaluation for one [`DiffOp`].
///
/// Binary ops call [`resolve_ref`], which may grow `leaf_stack`.
fn step_diff(
    t: Tensor<DiffB, 2>,
    op: &DiffOp,
    leaf_stack: &mut Vec<Tensor<DiffB, 2>>,
    leaf_data: &[Vec<u8>],
    max_leaves: usize,
    rows: usize,
    cols: usize,
    device: &NdArrayDevice,
) -> Tensor<DiffB, 2> {
    match op {
        DiffOp::Add(r) => {
            let rhs = resolve_ref(r, &t, leaf_stack, leaf_data, max_leaves, rows, cols, device);
            t.clone() + rhs
        }
        DiffOp::Sub(r) => {
            let rhs = resolve_ref(r, &t, leaf_stack, leaf_data, max_leaves, rows, cols, device);
            t.clone() - rhs
        }
        DiffOp::Mul(r) => {
            let rhs = resolve_ref(r, &t, leaf_stack, leaf_data, max_leaves, rows, cols, device);
            t.clone() * rhs
        }
        DiffOp::Neg     => t.neg(),
        DiffOp::Exp     => t.exp(),
        DiffOp::Log     => t.log(),
        DiffOp::Sqrt    => t.sqrt(),
        DiffOp::Sigmoid => activation::sigmoid(t),
        DiffOp::Tanh    => activation::tanh(t),
        DiffOp::SumAll  => t.sum().unsqueeze::<2>(),
        DiffOp::Clamp   => t.clamp(-1e6_f32, 1e6_f32),
    }
}
