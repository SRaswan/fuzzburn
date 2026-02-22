// IR interpreter – walks the AST and runs the Burn tensor API.

use std::panic;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{activation, Tensor};
use burn::tensor::backend::{AutodiffBackend, Backend};

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use super::ops::{DiffOp, TensorOp, TensorRef};
use super::program::{AutogradProgram, FuzzConfig, HarnessMode, SingleOpCase, TensorProgram};

type PlainB = NdArray;
type DiffB = Autodiff<NdArray>;

/// Cycle raw bytes and map to f32 values in [-1, 1]
fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() {
        return vec![0.5_f32; n];
    }
    (0..n)
        .map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0)
        .collect()
}

/// - `PanicOnFirstError` -> resume unwind and libFuzzer records crash
/// - `Continuous`        -> log to stderr and return (fuzzer keeps running)
fn handle_crash(e: Box<dyn std::any::Any + Send>, display: &str, target: &str, mode: HarnessMode) {
    eprintln!("\n=== CRASH DETECTED ({target}) ===");
    eprintln!("{display}");
    eprintln!("========================================\n");
    match mode {
        HarnessMode::PanicOnFirstError => panic::resume_unwind(e),
        HarnessMode::Continuous => {  }
    }
}

// TENSOR OPS
// ─────────────────────────────────────────────────────────────────────
/// Build a 2-D tensor from raw seed bytes for any backend.
fn make_plain_tensor<B: Backend>(raw: &[u8], rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(bytes_to_floats(raw, rows * cols).as_slice(), device)
        .reshape([rows, cols])
}

/// Apply one TensorOp to `lhs` and `rhs`
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

// SingleOP Interpreter
// ───────────────────────────────────────────────────────────────────
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

/// run on Libtorch
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

/// TODO: make tolerance an environmental var
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

/// compare with libtorch
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

// plain tensor interpreter 
// ─────────────────────────────────────────────────

/// run TensorProgram against plain NdArray backend
/// TODO: compare with libtorch
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

/// Single-step evaluation for one TensorOp
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

// Autograd interpreter 
// ─────────────────────────────────────────────────────

fn make_leaf<AB: AutodiffBackend>(raw: &[u8], rows: usize, cols: usize, device: &AB::Device) -> Tensor<AB, 2> {
    Tensor::<AB, 1>::from_floats(
        bytes_to_floats(raw, rows * cols).as_slice(),
        device,
    )
    .reshape([rows, cols])
    .require_grad()
}

/// Resolve a [`TensorRef`] using the leaf-stack dispatch algorithm.
///
/// - `TensorRef::Current` → clone the accumulator t
/// - `TensorRef::Leaf(raw)` → reuse or introduce a leaf via
///   [`TensorRef::dispatch_leaf_idx`], growing `leaf_stack` when a new leaf used
///   is introduced.
fn resolve_ref<AB: AutodiffBackend>(
    r: &TensorRef,
    t: &Tensor<AB, 2>,
    leaf_stack: &mut Vec<Tensor<AB, 2>>,
    leaf_data: &[Vec<u8>],
    max_leaves: usize,
    rows: usize,
    cols: usize,
    device: &AB::Device,
) -> Tensor<AB, 2> {
    match r.dispatch_leaf_idx(leaf_stack.len(), max_leaves) {
        None => t.clone(), // TensorRef::Current
        Some((idx, new_count)) => {
            if new_count > leaf_stack.len() {
                // Introduce a new leaf at position `idx` (= old stack length).
                let raw = leaf_data.get(idx).map(Vec::as_slice).unwrap_or(&[]);
                leaf_stack.push(make_leaf::<AB>(raw, rows, cols, device));
            }
            leaf_stack[idx].clone()
        }
    }
}

/// Single-step evaluation for one [`DiffOp`].
///
/// Binary ops call [`resolve_ref`], which may grow `leaf_stack`.
fn step_diff<AB: AutodiffBackend>(
    t: Tensor<AB, 2>,
    op: &DiffOp,
    leaf_stack: &mut Vec<Tensor<AB, 2>>,
    leaf_data: &[Vec<u8>],
    max_leaves: usize,
    rows: usize,
    cols: usize,
    device: &AB::Device,
) -> Tensor<AB, 2> {
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

/// Run `prog` on any autodiff backend, returning gradient data for every leaf
/// in stack order.  Panics if any leaf is missing a gradient.
fn collect_grads<AB: AutodiffBackend>(
    prog: &AutogradProgram,
    config: &FuzzConfig,
    device: &AB::Device,
) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    let leaf_0 = make_leaf::<AB>(prog.leaves.get(0).map(Vec::as_slice).unwrap_or(&[]), rows, cols, device);
    let mut leaf_stack: Vec<Tensor<AB, 2>> = vec![leaf_0.clone()];
    let mut t = leaf_0;
    for op in &prog.ops {
        t = step_diff(t, op, &mut leaf_stack, &prog.leaves, config.max_leaves, rows, cols, device);
    }
    let grads = t.backward();
    leaf_stack.iter().enumerate().map(|(i, leaf)| {
        leaf.grad(&grads)
            .unwrap_or_else(|| panic!("grad of leaf x_{i} missing after backward()"))
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|e| panic!("into_data for x_{i} grad failed: {e}"))
    }).collect()
}

/// run AutogradProgram against Autodiff<NdArray> backend
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) {
    let display = prog.ssa(config.max_leaves);
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        collect_grads::<DiffB>(prog, config, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        run_autograd_oracle(prog, config);
    }));
    if let Err(e) = result {
        handle_crash(e, &display, "fuzz_autograd", config.mode);
    }
}

// ─── autograd oracle (feature = "oracle-tch") ─────────────────────────────────────────────────

/// The LibTorch autodiff backend type alias, mirroring `DiffB` for NdArray.
#[cfg(feature = "oracle-tch")]
type DiffTch = Autodiff<LibTorch>;

/// Compare autograd outputs between NdArray and LibTorch.
///
/// Both backends run the same [`AutogradProgram`] with identical leaf seeds.
/// A mismatch in leaf count means one backend introduced a leaf the other
/// didn't — that is itself a bug.  Value mismatches use the same
/// relative+absolute tolerance as the single-op oracle (`1e-4 × scale`).
#[cfg(feature = "oracle-tch")]
fn run_autograd_oracle(prog: &AutogradProgram, config: &FuzzConfig) {
    let nd = collect_grads::<DiffB>(prog, config, &NdArrayDevice::default());
    let lt = collect_grads::<DiffTch>(prog, config, &LibTorchDevice::Cpu);
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
