//! IR interpreter – walks SSA programs using a register-file architecture.
//!
//! All public entry-points return `Result<(), String>`.  The fuzz target
//! decides what to do: in `PanicOnFirstError` mode it panics (so libFuzzer
//! saves the crash artifact), in `Continuous` it logs to stderr and moves on.

mod tensor_program;
mod autograd;

pub use tensor_program::run_tensor_program;
pub use autograd::run_autograd_program;

use burn::backend::NdArray;
use burn::tensor::{activation, Tensor};
use burn::tensor::backend::Backend;

use super::ops::{Reg, TensorInstr};

type PlainB = NdArray;

// ─── shared utilities ────────────────────────────────────────────────────────

/// Cycle raw bytes and map to f32 values in [-1, 1].
fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() {
        return vec![0.5_f32; n];
    }
    (0..n)
        .map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0)
        .collect()
}

/// Wrap a closure that may panic, converting the panic into `Err(String)`.
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

/// Compare two output vectors element-wise with relative tolerance.
#[cfg(feature = "oracle-tch")]
fn compare_outputs(ndarray: &[f32], libtorch: &[f32], label: &str) {
    assert_eq!(ndarray.len(), libtorch.len(), "oracle shape mismatch in {label}");
    let mut mismatches = 0_usize;
    for (_i, (&a, &b)) in ndarray.iter().zip(libtorch).enumerate() {
        if a.is_nan() && b.is_nan() { continue; }
        let abs_diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0_f32);
        if abs_diff > 1e-4_f32 * scale {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        panic!("oracle detected {} mismatches in {}", mismatches, label);
    }
}

// ─── shared instruction evaluator ────────────────────────────────────────────

/// Evaluate one [`TensorInstr`] against the register file.
/// The generator guarantees every `Reg` is a valid index.
pub(super) fn eval_tensor_instr_direct<B: Backend>(
    regs: &[Tensor<B, 2>],
    instr: &TensorInstr,
) -> Tensor<B, 2> {
    let r = |reg: &Reg| -> Tensor<B, 2> { regs[usize::from(*reg)].clone() };
    match instr {
        TensorInstr::Add(a, b)    => r(a) + r(b),
        TensorInstr::Sub(a, b)    => r(a) - r(b),
        TensorInstr::Mul(a, b)    => r(a) * r(b),
        TensorInstr::Div(a, b)    => r(a) / r(b),
        TensorInstr::Matmul(a, b) => r(a).matmul(r(b)),
        TensorInstr::Neg(x)       => r(x).neg(),
        TensorInstr::Abs(x)       => r(x).abs(),
        TensorInstr::Exp(x)       => r(x).exp(),
        TensorInstr::Log(x)       => r(x).log(),
        TensorInstr::Sqrt(x)      => r(x).sqrt(),
        TensorInstr::Cos(x)       => r(x).cos(),
        TensorInstr::Sin(x)       => r(x).sin(),
        TensorInstr::Relu(x)      => activation::relu(r(x)),
        TensorInstr::Sigmoid(x)   => activation::sigmoid(r(x)),
        TensorInstr::Tanh(x)      => activation::tanh(r(x)),
        TensorInstr::SumAll(x)    => r(x).sum().unsqueeze::<2>(),
        TensorInstr::MeanAll(x)   => r(x).mean().unsqueeze::<2>(),
        TensorInstr::SumDim(x, d)  => { let dim = *d as usize % 2; r(x).sum_dim(dim) }
        TensorInstr::MeanDim(x, d) => { let dim = *d as usize % 2; r(x).mean_dim(dim) }
        TensorInstr::ArgMax(x, d)  => { let dim = *d as usize % 2; r(x).argmax(dim).float() }
        TensorInstr::Transpose(x) => r(x).transpose(),
        TensorInstr::Concat(a, b, d) => {
            let dim = *d as usize % 2;
            Tensor::cat(vec![r(a), r(b)], dim)
        }
        TensorInstr::Repeat(x, d, c) => {
            let dim = *d as usize % 2;
            r(x).repeat_dim(dim, (*c as usize).clamp(1, 4))
        }
        TensorInstr::Slice(x, d, len) => {
            let t = r(x);
            let dims = t.dims();
            let dim = *d as usize % 2;
            let take = (*len as usize).clamp(1, dims[dim]);
            if dim == 0 { t.slice([0..take, 0..dims[1]]) }
            else { t.slice([0..dims[0], 0..take]) }
        }
        TensorInstr::Powf(x, c) => {
            const EXP: [f32; 5] = [0.5, 2.0, 3.0, -1.0, 1.5];
            r(x).powf_scalar(EXP[(*c as usize) % EXP.len()])
        }
        TensorInstr::Clamp(x)     => r(x).clamp(-1e6_f32, 1e6_f32),
    }
}
