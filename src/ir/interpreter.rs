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
/// `config.num_inputs` controls how many leaf tensors are built (1–8).
/// `config.mode` controls crash behaviour (re-panic vs. continuous logging).
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) {
    let display = prog.ssa(config.num_inputs);
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
    let n = config.num_inputs; // already clamped to [1, 8] by FuzzConfig::from_env
    let device = NdArrayDevice::default();

    // Build N leaf tensors.  If `prog.leaves` has fewer entries than `n` we
    // fall back to an empty slice so `bytes_to_floats` uses the 0.5 default.
    let leaves: Vec<Tensor<DiffB, 2>> = (0..n)
        .map(|i| {
            let raw = prog.leaves.get(i).map(Vec::as_slice).unwrap_or(&[]);
            Tensor::<DiffB, 1>::from_floats(
                bytes_to_floats(raw, rows * cols).as_slice(),
                &device,
            )
            .reshape([rows, cols])
            .require_grad()
        })
        .collect();

    // Accumulator starts as the element-wise sum of all leaves so every leaf
    // is in the compute graph even when `prog.ops` is empty.
    let mut t: Tensor<DiffB, 2> = leaves
        .iter()
        .skip(1)
        .fold(leaves[0].clone(), |acc, x| acc + x.clone());

    for op in &prog.ops {
        t = step_diff(t, &leaves, op);
    }

    let loss = t.sum();
    let grads = loss.backward();

    // Correctness assertion: every leaf must have a gradient.
    for (i, leaf) in leaves.iter().enumerate() {
        if leaf.grad(&grads).is_none() {
            panic!("grad of leaf x_{i} missing after backward()");
        }
    }
}

/// Resolve a [`TensorRef`] to a concrete tensor.
///
/// - `Leaf(i)` → `leaves[i as usize % leaves.len()]`
/// - `Current` → current accumulator `t`
fn resolve(t: &Tensor<DiffB, 2>, leaves: &[Tensor<DiffB, 2>], r: &TensorRef) -> Tensor<DiffB, 2> {
    match r {
        TensorRef::Leaf(i) => leaves[(*i as usize) % leaves.len()].clone(),
        TensorRef::Current => t.clone(),
    }
}

/// Single-step evaluation for one [`DiffOp`].
fn step_diff(
    t: Tensor<DiffB, 2>,
    leaves: &[Tensor<DiffB, 2>],
    op: &DiffOp,
) -> Tensor<DiffB, 2> {
    match op {
        DiffOp::Add(r) => t.clone() + resolve(&t, leaves, r),
        DiffOp::Sub(r) => t.clone() - resolve(&t, leaves, r),
        DiffOp::Mul(r) => t.clone() * resolve(&t, leaves, r),
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
