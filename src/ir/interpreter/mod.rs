//! IR interpreter – walks SSA programs using a register-file architecture.

mod tensor_program;
mod autograd;

pub use tensor_program::run_tensor_program;
pub use autograd::run_autograd_program;

use burn::backend::NdArray;
use burn::tensor::{activation, Tensor};
use burn::tensor::backend::Backend;

use crate::ir::program::FuzzConfig;
use crate::ir::ops::{Reg, TensorInstr};

type PlainB = NdArray;

// ─── shared utilities ────────────────────────────────────────────────────────

fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() {
        return vec![0.5_f32; n];
    }
    (0..n)
        .map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0)
        .collect()
}

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

#[cfg(feature = "oracle-tch")]
fn compare_outputs(ndarray: &[f32], libtorch: &[f32], label: &str) {
    assert_eq!(ndarray.len(), libtorch.len(), "oracle shape mismatch in {label}");
    let mut mismatches = 0usize;
    for (&a, &b) in ndarray.iter().zip(libtorch) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        let abs_diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        if abs_diff > 1e-4 * scale {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        panic!("oracle detected {} mismatches in {}", mismatches, label);
    }
}

#[inline]
fn idx(r: &Reg, n: usize) -> usize {
    (r.0 as usize) % n
}

/// Direct evaluator: assumes the generator has already ensured operands are legal.
fn eval_tensor_instr_direct<B: Backend>(
    regs: &[Tensor<B, 2>],
    instr: &TensorInstr,
    config: &FuzzConfig,
) -> Tensor<B, 2> {
    let n = regs.len();

    match instr {
        TensorInstr::Add(a, b) => regs[idx(a, n)].clone() + regs[idx(b, n)].clone(),
        TensorInstr::Sub(a, b) => regs[idx(a, n)].clone() - regs[idx(b, n)].clone(),
        TensorInstr::Mul(a, b) => regs[idx(a, n)].clone() * regs[idx(b, n)].clone(),
        TensorInstr::Div(a, b) => {
            let a = regs[idx(a, n)].clone();
            let b = regs[idx(b, n)].clone();
            if config.safe_math {
                a / b.clamp(1e-6, 1e6)
            } else {
                a / b
            }
        }

        TensorInstr::Matmul(a, b) => regs[idx(a, n)].clone().matmul(regs[idx(b, n)].clone()),

        TensorInstr::Neg(r) => regs[idx(r, n)].clone().neg(),
        TensorInstr::Abs(r) => regs[idx(r, n)].clone().abs(),
        TensorInstr::Exp(r) => regs[idx(r, n)].clone().exp(),
        TensorInstr::Cos(r) => regs[idx(r, n)].clone().cos(),
        TensorInstr::Sin(r) => regs[idx(r, n)].clone().sin(),

        TensorInstr::Log(r) => {
            let x = regs[idx(r, n)].clone();
            if config.safe_math {
                x.clamp(1e-6, 1e6).log()
            } else {
                x.log()
            }
        }

        TensorInstr::Sqrt(r) => {
            let x = regs[idx(r, n)].clone();
            if config.safe_math {
                x.clamp(0.0, 1e6).sqrt()
            } else {
                x.sqrt()
            }
        }

        // TensorInstr::Powf(r, c) => {
        //     // if your Powf stores exponent as u8
        //     let x = regs[idx(r, n)].clone();
        //     x.powf((*c as f32).clamp(-8.0, 8.0))
        // }

        TensorInstr::Powf(r, c) => {
            let x = regs[idx(r, n)].clone();

            // exponent as a scalar tensor, then broadcast to x's shape
            let exp: f32 = (*c as f32).clamp(-8.0, 8.0);
            let dev = x.device();

            let e = Tensor::<B, 1>::from_floats([exp].as_slice(), &dev)
                .unsqueeze::<2>()                 // [1, 1]
                .expand(x.dims());                // [rows, cols]

            x.powf(e)
        }

        TensorInstr::Relu(r) => activation::relu(regs[idx(r, n)].clone()),
        TensorInstr::Sigmoid(r) => activation::sigmoid(regs[idx(r, n)].clone()),
        TensorInstr::Tanh(r) => activation::tanh(regs[idx(r, n)].clone()),

        TensorInstr::SumAll(r) => regs[idx(r, n)].clone().sum().unsqueeze::<2>(),
        TensorInstr::MeanAll(r) => regs[idx(r, n)].clone().mean().unsqueeze::<2>(),

        TensorInstr::SumDim(r, d) => regs[idx(r, n)].clone().sum_dim((*d as usize) % 2),
        TensorInstr::MeanDim(r, d) => regs[idx(r, n)].clone().mean_dim((*d as usize) % 2),

        TensorInstr::ArgMax(r, d) => {
            // keep shape consistent with your generator’s expected output:
            // Burn argmax usually returns integer tensor; easiest is to just no-op
            // (still deterministic and safe for fuzzing shape legality).
            let _dim = (*d as usize) % 2;
            regs[idx(r, n)].clone()
        }

        TensorInstr::Transpose(r) => regs[idx(r, n)].clone().transpose(),

        TensorInstr::Concat(a, b, d) => {
            let dim = (*d as usize) % 2;
            Tensor::cat(vec![regs[idx(a, n)].clone(), regs[idx(b, n)].clone()], dim)
        }

        TensorInstr::Repeat(r, d, c) => {
            let dim = (*d as usize) % 2;
            let count = (*c as usize).clamp(1, 4);
            regs[idx(r, n)].clone().repeat_dim(dim, count)
        }

        TensorInstr::Slice(r, d, len) => {
            // basic “prefix slice” semantics; keeps things deterministic.
            let dim = (*d as usize) % 2;
            let t = regs[idx(r, n)].clone();
            let [rows, cols] = t.dims();
            let take = (*len as usize).clamp(1, if dim == 0 { rows } else { cols });

            if dim == 0 {
                t.slice([0..take, 0..cols])
            } else {
                t.slice([0..rows, 0..take])
            }
        }

        TensorInstr::Clamp(r) => regs[idx(r, n)].clone().clamp(-1e6, 1e6),
    }
}