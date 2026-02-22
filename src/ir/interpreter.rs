// IR interpreter – walks SSA programs using a register-file architecture.
//
// All public entry-points return `Result<(), String>`.  The fuzz target
// decides what to do: in `PanicOnFirstError` mode it panics (so libFuzzer
// saves the crash artifact), in `Continuous` it logs to stderr and moves on.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{activation, Tensor};
use burn::tensor::backend::{AutodiffBackend, Backend};

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use super::ops::{DiffOp, TensorOp, TensorInstr};
use super::program::{AutogradProgram, FuzzConfig, SingleOpCase, TensorProgram};

type PlainB = NdArray;
type DiffB = Autodiff<NdArray>;

// ─── shape tracking ──────────────────────────────────────────────────────────

/// Lightweight 2-D shape tracked alongside every register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape2(usize, usize);

impl Shape2 {
    /// Can two 2-D shapes be element-wise combined under NumPy broadcasting?
    /// Each dimension must either be equal or one of them must be 1.
    #[inline]
    fn broadcast_compatible(self, other: Shape2) -> bool {
        (self.0 == other.0 || self.0 == 1 || other.0 == 1)
            && (self.1 == other.1 || self.1 == 1 || other.1 == 1)
    }

    /// Result shape after broadcasting two compatible shapes.
    #[inline]
    fn broadcast_result(self, other: Shape2) -> Shape2 {
        Shape2(self.0.max(other.0), self.1.max(other.1))
    }

    /// Can `self` be left-multiplied by `other`?  i.e. `self @ other`
    /// requires self.cols == other.rows.
    #[inline]
    fn matmul_compatible(self, other: Shape2) -> bool {
        self.1 == other.0
    }

    /// Given a unary/binary TensorInstr, compute the output shape.
    fn after_tensor_instr(shapes: &[Shape2], instr: &TensorInstr) -> Shape2 {
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
    fn after_diff_op(shapes: &[Shape2], op: &DiffOp) -> Option<Shape2> {
        match op {
            DiffOp::Leaf { .. } => None,
            DiffOp::Instr(instr) => Some(Shape2::after_tensor_instr(shapes, instr)),
        }
    }
}

/// For element-wise binary operations (Add/Sub/Mul), resolve operand `b` to a
/// register whose shape is **broadcast-compatible** with operand `a`.
/// Falls back to `a` itself when nothing compatible exists.
fn resolve_broadcast_compatible(shapes: &[Shape2], a_idx: usize, b_raw: &super::ops::Reg) -> usize {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.broadcast_compatible(shapes[b_idx]) {
        return b_idx;
    }
    // scan backward for any compatible register
    for i in (0..n).rev() {
        if sa.broadcast_compatible(shapes[i]) {
            return i;
        }
    }
    a_idx // ultimate fallback: a ⊕ a
}

/// For matmul, resolve operand `b` to a register where
/// `shapes[a_idx].cols == shapes[b_idx].rows`.
/// Returns `None` when no register in the file has compatible inner
/// dimensions — callers should demote the matmul to a passthrough.
fn resolve_matmul_compatible(shapes: &[Shape2], a_idx: usize, b_raw: &super::ops::Reg) -> Option<usize> {
    let n = shapes.len();
    let b_idx = b_raw.resolve(n);
    let sa = shapes[a_idx];
    if sa.matmul_compatible(shapes[b_idx]) {
        return Some(b_idx);
    }
    // scan backward for any register whose rows == a.cols
    for i in (0..n).rev() {
        if sa.matmul_compatible(shapes[i]) {
            return Some(i);
        }
    }
    None // no valid matmul partner exists
}

/// Cycle raw bytes and map to f32 values in [-1, 1]
fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() {
        return vec![0.5_f32; n];
    }
    (0..n)
        .map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0)
        .collect()
}

/// Wrap a closure that may panic, converting the panic into `Err(String)`.
/// This swaps in a no-op panic hook so libfuzzer's aborting hook doesn't
/// kill us, then restores the original hook afterwards.
fn catch_as_result<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> Result<(), String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    result.map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "<unknown panic payload>".to_string()
        }
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// SINGLE-OP (unchanged – operates on two explicit tensors, not an SSA program)
// ═══════════════════════════════════════════════════════════════════════════════

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
        TensorOp::Matmul    => lhs.matmul(rhs),
        TensorOp::Clamp     => lhs.clamp(-1e6_f32, 1e6_f32),
    }
}

// SingleOP Interpreter
// ───────────────────────────────────────────────────────────────────
pub fn run_single_op_case(case: &SingleOpCase) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        run_single_op_case_inner(case);
        #[cfg(feature = "oracle-tch")]
        run_single_op_oracle(case);
    }))
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
fn compare_outputs(ndarray: &[f32], libtorch: &[f32], label: &str){
    assert_eq!(ndarray.len(), libtorch.len(), "oracle shape mismatch in {label}");
    let mut mismatches = 0_usize;
    for (i, (&a, &b)) in ndarray.iter().zip(libtorch).enumerate() {
        // Both NaN → agree (NaN propagation is correct on both backends).
        if a.is_nan() && b.is_nan() { continue; }
        let abs_diff = (a - b).abs();
        let scale    = a.abs().max(b.abs()).max(1.0_f32);
        if abs_diff > 1e-4_f32 * scale {
            // eprintln!("oracle mismatch [{i}]: NdArray={a}, LibTorch={b}  ({label})");
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        panic!("oracle detected {} mismatches in {}", mismatches, label);
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

// ═══════════════════════════════════════════════════════════════════════════════
// PLAIN TENSOR PROGRAM  (SSA register-file interpreter)
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluate one [`TensorInstr`] against the register file, using `shapes`
/// to ensure binary operands are shape-compatible.
fn eval_tensor_instr<B: Backend>(
    regs: &[Tensor<B, 2>],
    shapes: &[Shape2],
    instr: &TensorInstr,
) -> Tensor<B, 2> {
    let n = regs.len();
    match instr {
        TensorInstr::Add(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() + regs[bi].clone()
        }
        TensorInstr::Sub(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() - regs[bi].clone()
        }
        TensorInstr::Mul(a, b) => {
            let ai = a.resolve(n);
            let bi = resolve_broadcast_compatible(shapes, ai, b);
            regs[ai].clone() * regs[bi].clone()
        }
        TensorInstr::Matmul(a, b) => {
            let ai = a.resolve(n);
            match resolve_matmul_compatible(shapes, ai, b) {
                Some(bi) => regs[ai].clone().matmul(regs[bi].clone()),
                None => regs[ai].clone(), // no valid partner → passthrough
            }
        }
        TensorInstr::Neg(r)       => regs[r.resolve(n)].clone().neg(),
        TensorInstr::Abs(r)       => regs[r.resolve(n)].clone().abs(),
        TensorInstr::Exp(r)       => regs[r.resolve(n)].clone().exp(),
        TensorInstr::Log(r)       => regs[r.resolve(n)].clone().log(),
        TensorInstr::Sqrt(r)      => regs[r.resolve(n)].clone().sqrt(),
        TensorInstr::Relu(r)      => activation::relu(regs[r.resolve(n)].clone()),
        TensorInstr::Sigmoid(r)   => activation::sigmoid(regs[r.resolve(n)].clone()),
        TensorInstr::Tanh(r)      => activation::tanh(regs[r.resolve(n)].clone()),
        TensorInstr::SumAll(r)    => regs[r.resolve(n)].clone().sum().unsqueeze::<2>(),
        TensorInstr::MeanAll(r)   => regs[r.resolve(n)].clone().mean().unsqueeze::<2>(),
        TensorInstr::Transpose(r) => regs[r.resolve(n)].clone().transpose(),
        TensorInstr::Clamp(r)     => regs[r.resolve(n)].clone().clamp(-1e6_f32, 1e6_f32),
    }
}

/// Run a plain SSA TensorProgram on any backend, returning the final
/// register's data as `Vec<f32>` (mirrors the `collect_grads` pattern).
fn eval_tensor_program<B: Backend>(prog: &TensorProgram, device: &B::Device) -> Vec<f32> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    // r0 = initial tensor
    let r0: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(
            bytes_to_floats(&prog.values, rows * cols).as_slice(),
            device,
        )
        .reshape([rows, cols]);

    let mut regs: Vec<Tensor<B, 2>> = vec![r0];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];

    for instr in &prog.ops {
        let out_shape = Shape2::after_tensor_instr(&shapes, instr);
        let val = eval_tensor_instr(&regs, &shapes, instr);
        regs.push(val);
        shapes.push(out_shape);
    }

    regs.pop()
        .expect("register file is empty")
        .into_data()
        .to_vec::<f32>()
        .expect("into_data failed")
}

/// Run a plain SSA TensorProgram against NdArray (+ LibTorch oracle when enabled).
pub fn run_tensor_program(prog: &TensorProgram) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = eval_tensor_program::<PlainB>(prog, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        run_tensor_program_oracle(prog, nd);
    }))
}

/// Compare tensor-program outputs between NdArray and LibTorch.
#[cfg(feature = "oracle-tch")]
fn run_tensor_program_oracle(prog: &TensorProgram, nd: Vec<f32>) {
    let lt = eval_tensor_program::<LibTorch>(prog, &LibTorchDevice::Cpu);
    compare_outputs(&nd, &lt, "tensor_program");
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOGRAD PROGRAM  (SSA register-file interpreter with gradient collection)
// ═══════════════════════════════════════════════════════════════════════════════

fn make_leaf<AB: AutodiffBackend>(raw: &[u8], rows: usize, cols: usize, device: &AB::Device) -> Tensor<AB, 2> {
    Tensor::<AB, 1>::from_floats(
        bytes_to_floats(raw, rows * cols).as_slice(),
        device,
    )
    .reshape([rows, cols])
    .require_grad()
}

/// Evaluate one non-Leaf [`DiffOp`] against the register file, using `shapes`
/// to ensure binary operands are shape-compatible.
/// Returns `None` for `Leaf` (handled by the main loop).
fn eval_diff_op<AB: AutodiffBackend>(
    regs: &[Tensor<AB, 2>],
    shapes: &[Shape2],
    op: &DiffOp,
) -> Option<Tensor<AB, 2>> {
    match op {
        DiffOp::Leaf { .. } => None,
        DiffOp::Instr(instr) => Some(eval_tensor_instr(regs, shapes, instr)),
    }
}

/// Run `prog` on any autodiff backend, returning gradient data for every leaf
/// in introduction order.  Panics if any leaf is missing a gradient.
fn collect_grads<AB: AutodiffBackend>(
    prog: &AutogradProgram,
    config: &FuzzConfig,
    device: &AB::Device,
) -> Vec<Vec<f32>> {
    let rows = (prog.rows as usize).clamp(1, 16);
    let cols = (prog.cols as usize).clamp(1, 16);

    // r0 = seed leaf (always present)
    let leaf_0 = make_leaf::<AB>(
        prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]),
        rows,
        cols,
        device,
    );
    let mut regs: Vec<Tensor<AB, 2>> = vec![leaf_0];
    let mut shapes: Vec<Shape2> = vec![Shape2(rows, cols)];
    let mut leaf_indices: Vec<usize> = vec![0]; // register indices that are leaves
    let mut leaf_shapes: Vec<(usize, usize)> = vec![(rows, cols)]; // per-leaf sizes
    let mut leaf_count: usize = 1;

    for op in &prog.ops {
        let (val, out_shape) = match op {
            DiffOp::Leaf { seed, rows: lr, cols: lc } => {
                if leaf_count < config.max_leaves {
                    let pool_idx = if prog.leaf_seeds.is_empty() {
                        0
                    } else {
                        *seed as usize % prog.leaf_seeds.len()
                    };
                    let raw = prog
                        .leaf_seeds
                        .get(pool_idx)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    let leaf_rows = (*lr as usize).clamp(1, 16);
                    let leaf_cols = (*lc as usize).clamp(1, 16);
                    let leaf = make_leaf::<AB>(raw, leaf_rows, leaf_cols, device);
                    leaf_indices.push(regs.len()); // index of the *next* push
                    leaf_shapes.push((leaf_rows, leaf_cols));
                    leaf_count += 1;
                    (leaf, Shape2(leaf_rows, leaf_cols))
                } else {
                    // Cap reached: alias an existing register
                    let alias_idx = (*seed as usize) % regs.len();
                    (regs[alias_idx].clone(), shapes[alias_idx])
                }
            }
            _ => {
                let out_shape = Shape2::after_diff_op(&shapes, op)
                    .expect("non-Leaf op returned None shape");
                let val = eval_diff_op(&regs, &shapes, op)
                    .expect("non-Leaf op returned None");
                (val, out_shape)
            }
        };
        regs.push(val);
        shapes.push(out_shape);
    }

    // backward from the last register
    let last = regs.last().expect("register file is empty").clone();
    let grads = last.backward();

    leaf_indices
        .iter()
        .zip(leaf_shapes.iter())
        .map(|(&ri, &(lr, lc))| {
            match regs[ri].grad(&grads) {
                Some(g) => g
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_else(|e| panic!("into_data for r{ri} grad failed: {e}")),
                // Leaf not reachable from the backward root → zero gradient
                None => vec![0.0_f32; lr * lc],
            }
        })
        .collect()
}

/// Run AutogradProgram against Autodiff<NdArray> backend.
pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = collect_grads::<DiffB>(prog, config, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        run_autograd_oracle(prog, config, nd);
    }))
}

// ─── autograd oracle (feature = "oracle-tch") ─────────────────────────────────

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
fn run_autograd_oracle(prog: &AutogradProgram, config: &FuzzConfig, nd: Vec<Vec<f32>>) {
    let lt = collect_grads::<DiffTch>(prog, config, &LibTorchDevice::Cpu);
    if nd.len() != lt.len() {
        panic!("oracle leaf count mismatch: NdArray={}, LibTorch={}", nd.len(), lt.len());
    }
    for (i, (nd_grad, lt_grad)) in nd.iter().zip(lt.iter()).enumerate() {
        compare_outputs(nd_grad, lt_grad, &format!("grad r{i}"));
    }
}
