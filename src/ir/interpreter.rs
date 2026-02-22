// IR interpreter – walks the AST and runs the Burn tensor API.

use std::panic;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{activation, Tensor};
use burn::tensor::backend::Backend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use super::ops::{DiffOp, TensorOp, TensorRef};
use super::program::{AutogradProgram, FuzzConfig, HarnessMode, SingleOpCase, TensorProgram};

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
// ─── generic tensor helpers ─────────────────────────────────────────────────────────────────────

/// Build a 2-D tensor from raw seed bytes for any backend.
fn make_plain_tensor<B: Backend>(raw: &[u8], rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(bytes_to_floats(raw, rows * cols).as_slice(), device)
        .reshape([rows, cols])
}

/// Apply one [`TensorOp`] to `lhs` (and `rhs` for binary ops) on any backend.
///
/// This is generic so the same match arms are used for both the NdArray
/// production run and the optional LibTorch oracle comparison.
fn apply_tensor_op<B: Backend>(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>, op: &TensorOp) -> Tensor<B, 2> {
    match op {
        TensorOp::Add       => lhs + rhs,
        TensorOp::Sub       => lhs - rhs,
        TensorOp::Mul       => lhs * rhs,
        TensorOp::Neg       => lhs.neg(),
        TensorOp::Abs       => lhs.abs(),
        TensorOp::Exp       => lhs.exp(),
        TensorOp::Log       => lhs.log(),
        TensorOp::Sqrt      => lhs.sqrt(),
        TensorOp::Relu      => activation::relu(lhs),
        TensorOp::Sigmoid   => activation::sigmoid(lhs),
        TensorOp::Tanh      => activation::tanh(lhs),
        TensorOp::SumAll    => lhs.sum().unsqueeze::<2>(),
        TensorOp::MeanAll   => lhs.mean().unsqueeze::<2>(),
        TensorOp::Transpose => lhs.transpose(),
        TensorOp::Clamp     => lhs.clamp(-1e6_f32, 1e6_f32),
    }
}

// ─── single-op interpreter ───────────────────────────────────────────────────────────────────

/// Execute a [`SingleOpCase`] against the NdArray backend.
///
/// Binary ops receive genuinely independent `lhs` / `rhs` tensors.
/// When compiled with `--features oracle-tch`, also runs on LibTorch and
/// compares outputs element-wise: any numerical divergence is a bug.
pub fn run_single_op_case(case: &SingleOpCase, mode: HarnessMode) {
    let display = case.to_string();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_single_op_case_inner(case);
        #[cfg(feature = "oracle-tch")]
        run_single_op_oracle(case);
    }));
    if let Err(e) = result {
        handle_crash(e, &display, "fuzz_tensor_ops", mode);
    }
}

fn run_single_op_case_inner(case: &SingleOpCase) {
    let rows = (case.rows as usize).clamp(1, 16);
    let cols = (case.cols as usize).clamp(1, 16);
    let device = NdArrayDevice::default();
    let lhs = make_plain_tensor::<NdArray>(&case.lhs, rows, cols, &device);
    let rhs = make_plain_tensor::<NdArray>(&case.rhs, rows, cols, &device);
    let result = apply_tensor_op::<NdArray>(lhs, rhs, &case.op);
    let _ = result.into_data(); // force evaluation
}

// ─── LibTorch oracle (only when feature = "oracle-tch") ──────────────────────────────────────────

/// Run `case` on the LibTorch backend, collect results as `Vec<f32>`.
#[cfg(feature = "oracle-tch")]
fn run_single_op_on_libtorch(case: &SingleOpCase) -> Vec<f32> {
    let rows = (case.rows as usize).clamp(1, 16);
    let cols = (case.cols as usize).clamp(1, 16);
    let device = LibTorchDevice::Cpu;
    let lhs = make_plain_tensor::<LibTorch>(&case.lhs, rows, cols, &device);
    let rhs = make_plain_tensor::<LibTorch>(&case.rhs, rows, cols, &device);
    apply_tensor_op::<LibTorch>(lhs, rhs, &case.op)
        .into_data()
        .to_vec::<f32>()
        .expect("oracle: into_data failed")
}

/// Compare NdArray and LibTorch outputs.  Tolerances match pyreking/cse291y:
/// abs(a−b) must be ≤ 1e-4 × max(|a|, |b|, 1.0).
#[cfg(feature = "oracle-tch")]
fn compare_outputs(ndarray: &[f32], libtorch: &[f32], label: &str) {
    assert_eq!(ndarray.len(), libtorch.len(), "oracle shape mismatch in {label}");
    let mut mismatches = 0_usize;
    for (i, (&a, &b)) in ndarray.iter().zip(libtorch).enumerate() {
        // Both NaN → agree (NaN propagation is correct on both backends).
        if a.is_nan() && b.is_nan() { continue; }
        let abs_diff = (a - b).abs();
        let scale    = a.abs().max(b.abs()).max(1.0_f32);
        if abs_diff > 1e-4_f32 * scale {
            eprintln!("oracle mismatch [{i}]: NdArray={a}, LibTorch={b}  ({label})");
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        panic!("{mismatches} oracle mismatch(es) in op {label}");
    }
}

/// Run a `SingleOpCase` on NdArray, run it again on LibTorch, compare.
#[cfg(feature = "oracle-tch")]
fn run_single_op_oracle(case: &SingleOpCase) {
    let rows = (case.rows as usize).clamp(1, 16);
    let cols = (case.cols as usize).clamp(1, 16);
    let device = NdArrayDevice::default();
    let lhs = make_plain_tensor::<NdArray>(&case.lhs, rows, cols, &device);
    let rhs = make_plain_tensor::<NdArray>(&case.rhs, rows, cols, &device);
    let ndarray_out = apply_tensor_op::<NdArray>(lhs, rhs, &case.op)
        .into_data()
        .to_vec::<f32>()
        .expect("ndarray: into_data failed");
    let libtorch_out = run_single_op_on_libtorch(case);
    compare_outputs(&ndarray_out, &libtorch_out, &format!("{:?}", case.op));
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
        // With oracle-tch: re-run on LibTorch, compare gradient values.
        #[cfg(feature = "oracle-tch")]
        run_autograd_oracle(prog, config);
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

// ─── autograd oracle (feature = "oracle-tch") ─────────────────────────────────────────────────

/// The LibTorch autodiff backend type alias, mirroring `DiffB` for NdArray.
#[cfg(feature = "oracle-tch")]
type DiffTch = Autodiff<LibTorch>;

/// Build one `requires_grad` leaf on the LibTorch backend.
#[cfg(feature = "oracle-tch")]
fn make_leaf_tch(raw: &[u8], rows: usize, cols: usize, device: &LibTorchDevice) -> Tensor<DiffTch, 2> {
    Tensor::<DiffTch, 1>::from_floats(
        bytes_to_floats(raw, rows * cols).as_slice(),
        device,
    )
    .reshape([rows, cols])
    .require_grad()
}

/// [`resolve_ref`] counterpart for the LibTorch backend.
#[cfg(feature = "oracle-tch")]
fn resolve_ref_tch(
    r: &TensorRef,
    t: &Tensor<DiffTch, 2>,
    leaf_stack: &mut Vec<Tensor<DiffTch, 2>>,
    leaf_data: &[Vec<u8>],
    max_leaves: usize,
    rows: usize,
    cols: usize,
    device: &LibTorchDevice,
) -> Tensor<DiffTch, 2> {
    match r.dispatch_leaf_idx(leaf_stack.len(), max_leaves) {
        None => t.clone(),
        Some((idx, new_count)) => {
            if new_count > leaf_stack.len() {
                let raw = leaf_data.get(idx).map(Vec::as_slice).unwrap_or(&[]);
                leaf_stack.push(make_leaf_tch(raw, rows, cols, device));
            }
            leaf_stack[idx].clone()
        }
    }
}

/// [`step_diff`] counterpart for the LibTorch backend.
#[cfg(feature = "oracle-tch")]
fn step_diff_tch(
    t: Tensor<DiffTch, 2>,
    op: &DiffOp,
    leaf_stack: &mut Vec<Tensor<DiffTch, 2>>,
    leaf_data: &[Vec<u8>],
    max_leaves: usize,
    rows: usize,
    cols: usize,
    device: &LibTorchDevice,
) -> Tensor<DiffTch, 2> {
    match op {
        DiffOp::Add(r) => {
            let rhs = resolve_ref_tch(r, &t, leaf_stack, leaf_data, max_leaves, rows, cols, device);
            t.clone() + rhs
        }
        DiffOp::Sub(r) => {
            let rhs = resolve_ref_tch(r, &t, leaf_stack, leaf_data, max_leaves, rows, cols, device);
            t.clone() - rhs
        }
        DiffOp::Mul(r) => {
            let rhs = resolve_ref_tch(r, &t, leaf_stack, leaf_data, max_leaves, rows, cols, device);
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

/// Run `prog` on NdArray, return gradient data for every leaf in stack order.
///
/// Panics if any leaf has no gradient (consistent with
/// [`run_autograd_program_inner`] — this will never be reached if the main
/// run already passed).
#[cfg(feature = "oracle-tch")]
fn collect_grads_ndarray(prog: &AutogradProgram, config: &FuzzConfig) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);
    let device = NdArrayDevice::default();

    let leaf_0 = make_leaf(prog.leaves.get(0).map(Vec::as_slice).unwrap_or(&[]), rows, cols, &device);
    let mut leaf_stack: Vec<Tensor<DiffB, 2>> = vec![leaf_0.clone()];
    let mut t = leaf_0;
    for op in &prog.ops {
        t = step_diff(t, op, &mut leaf_stack, &prog.leaves, config.max_leaves, rows, cols, &device);
    }
    let grads = t.sum().backward();
    leaf_stack.iter().enumerate().map(|(i, leaf)| {
        leaf.grad(&grads)
            .unwrap_or_else(|| panic!("oracle(ndarray): grad of x_{i} is None"))
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|e| panic!("oracle(ndarray): into_data for x_{i} grad failed: {e}"))
    }).collect()
}

/// Run `prog` on LibTorch, return gradient data for every leaf in stack order.
#[cfg(feature = "oracle-tch")]
fn collect_grads_libtorch(prog: &AutogradProgram, config: &FuzzConfig) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);
    let device = LibTorchDevice::Cpu;

    let leaf_0 = make_leaf_tch(prog.leaves.get(0).map(Vec::as_slice).unwrap_or(&[]), rows, cols, &device);
    let mut leaf_stack: Vec<Tensor<DiffTch, 2>> = vec![leaf_0.clone()];
    let mut t = leaf_0;
    for op in &prog.ops {
        t = step_diff_tch(t, op, &mut leaf_stack, &prog.leaves, config.max_leaves, rows, cols, &device);
    }
    let grads = t.sum().backward();
    leaf_stack.iter().enumerate().map(|(i, leaf)| {
        leaf.grad(&grads)
            .unwrap_or_else(|| panic!("oracle(libtorch): grad of x_{i} is None"))
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|e| panic!("oracle(libtorch): into_data for x_{i} grad failed: {e}"))
    }).collect()
}

/// Compare autograd outputs between NdArray and LibTorch.
///
/// Both backends run the same [`AutogradProgram`] with identical leaf seeds.
/// A mismatch in leaf count means one backend introduced a leaf the other
/// didn't — that is itself a bug.  Value mismatches use the same
/// relative+absolute tolerance as the single-op oracle (`1e-4 × scale`).
#[cfg(feature = "oracle-tch")]
fn run_autograd_oracle(prog: &AutogradProgram, config: &FuzzConfig) {
    let nd = collect_grads_ndarray(prog, config);
    let lt = collect_grads_libtorch(prog, config);
    if nd.len() != lt.len() {
        panic!(
            "autograd oracle: NdArray produced {} leaf gradients, LibTorch produced {}",
            nd.len(), lt.len()
        );
    }
    for (i, (nd_grad, lt_grad)) in nd.iter().zip(lt.iter()).enumerate() {
        compare_outputs(nd_grad, lt_grad, &format!("grad x_{i}"));
    }
}
