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

/// Convert raw bytes into floats in [-1,1], cycling bytes if needed.
pub(crate) fn bytes_to_floats(raw: &[u8], n: usize) -> Vec<f32> {
    if raw.is_empty() { return vec![0.5; n]; }
    (0..n).map(|i| raw[i % raw.len()] as f32 / 128.0 - 1.0).collect()
}

/// Catch panics and return them as Err(String).
pub(crate) fn catch_as_result<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> Result<(), String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    r.map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() { (*s).to_string() }
        else if let Some(s) = e.downcast_ref::<String>() { s.clone() }
        else { "<unknown panic payload>".to_string() }
    })
}

#[cfg(feature = "oracle-tch")]
pub(crate) fn compare_outputs(nd: &[f32], lt: &[f32], label: &str, cfg: &FuzzConfig) {
    assert_eq!(nd.len(), lt.len(), "oracle shape mismatch in {label}");
    let mut mismatches = 0usize;
    for (i, (&a, &b)) in nd.iter().zip(lt).enumerate() {
        if a.is_nan() && b.is_nan() { continue; }
        let abs_diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        if abs_diff > cfg.tol_abs.max(cfg.tol_rel * scale) {
            mismatches += 1;
            if mismatches <= 8 {
                eprintln!("oracle mismatch [{i}]: NdArray={a}, LibTorch={b} ({label})");
            }
        }
        if mismatches >= cfg.max_mismatches { break; }
    }
    if mismatches > 0 {
        panic!("oracle detected {} mismatches in {}", mismatches, label);
    }
}

#[inline]
fn idx(r: &Reg, n: usize) -> usize { (r.0 as usize) % n }

#[derive(Clone)]
pub enum AnyTensor<B: Backend> {
    T1(Tensor<B, 1>),
    T2(Tensor<B, 2>),
    T3(Tensor<B, 3>),
    T4(Tensor<B, 4>),
}

impl<B: Backend> AnyTensor<B> {
    pub fn rank(&self) -> u8 {
        match self { AnyTensor::T1(_) => 1, AnyTensor::T2(_) => 2, AnyTensor::T3(_) => 3, AnyTensor::T4(_) => 4 }
    }

    pub fn to_vec_f32(self) -> Vec<f32> {
        match self {
            AnyTensor::T1(t) => t.into_data().to_vec::<f32>().unwrap_or_default(),
            AnyTensor::T2(t) => t.into_data().to_vec::<f32>().unwrap_or_default(),
            AnyTensor::T3(t) => t.into_data().to_vec::<f32>().unwrap_or_default(),
            AnyTensor::T4(t) => t.into_data().to_vec::<f32>().unwrap_or_default(),
        }
    }

    fn passthrough(&self) -> AnyTensor<B> { self.clone() }
}

/// Direct evaluator. If an op isn’t implemented for a given rank, it falls back
/// to passthrough (keeps fuzzing stable while you expand rank support).
pub(crate) fn eval_tensor_instr_direct<B: Backend>(
    regs: &[AnyTensor<B>],
    instr: &TensorInstr,
    cfg: &FuzzConfig,
) -> AnyTensor<B> {
    let n = regs.len();
    match instr {
        TensorInstr::Add(a,b) => match (&regs[idx(a,n)], &regs[idx(b,n)]) {
            (AnyTensor::T1(x), AnyTensor::T1(y)) => AnyTensor::T1(x.clone() + y.clone()),
            (AnyTensor::T2(x), AnyTensor::T2(y)) => AnyTensor::T2(x.clone() + y.clone()),
            (AnyTensor::T3(x), AnyTensor::T3(y)) => AnyTensor::T3(x.clone() + y.clone()),
            (AnyTensor::T4(x), AnyTensor::T4(y)) => AnyTensor::T4(x.clone() + y.clone()),
            _ => regs[idx(a,n)].passthrough(),
        },
        TensorInstr::Sub(a,b) => match (&regs[idx(a,n)], &regs[idx(b,n)]) {
            (AnyTensor::T1(x), AnyTensor::T1(y)) => AnyTensor::T1(x.clone() - y.clone()),
            (AnyTensor::T2(x), AnyTensor::T2(y)) => AnyTensor::T2(x.clone() - y.clone()),
            (AnyTensor::T3(x), AnyTensor::T3(y)) => AnyTensor::T3(x.clone() - y.clone()),
            (AnyTensor::T4(x), AnyTensor::T4(y)) => AnyTensor::T4(x.clone() - y.clone()),
            _ => regs[idx(a,n)].passthrough(),
        },
        TensorInstr::Mul(a,b) => match (&regs[idx(a,n)], &regs[idx(b,n)]) {
            (AnyTensor::T1(x), AnyTensor::T1(y)) => AnyTensor::T1(x.clone() * y.clone()),
            (AnyTensor::T2(x), AnyTensor::T2(y)) => AnyTensor::T2(x.clone() * y.clone()),
            (AnyTensor::T3(x), AnyTensor::T3(y)) => AnyTensor::T3(x.clone() * y.clone()),
            (AnyTensor::T4(x), AnyTensor::T4(y)) => AnyTensor::T4(x.clone() * y.clone()),
            _ => regs[idx(a,n)].passthrough(),
        },
        TensorInstr::Div(a,b) => match (&regs[idx(a,n)], &regs[idx(b,n)]) {
            (AnyTensor::T1(x), AnyTensor::T1(y)) => {
                let denom = if cfg.safe_math { y.clone().clamp(1e-6, 1e6) } else { y.clone() };
                AnyTensor::T1(x.clone() / denom)
            }
            (AnyTensor::T2(x), AnyTensor::T2(y)) => {
                let denom = if cfg.safe_math { y.clone().clamp(1e-6, 1e6) } else { y.clone() };
                AnyTensor::T2(x.clone() / denom)
            }
            (AnyTensor::T3(x), AnyTensor::T3(y)) => {
                let denom = if cfg.safe_math { y.clone().clamp(1e-6, 1e6) } else { y.clone() };
                AnyTensor::T3(x.clone() / denom)
            }
            (AnyTensor::T4(x), AnyTensor::T4(y)) => {
                let denom = if cfg.safe_math { y.clone().clamp(1e-6, 1e6) } else { y.clone() };
                AnyTensor::T4(x.clone() / denom)
            }
            _ => regs[idx(a,n)].passthrough(),
        },

        TensorInstr::Neg(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(x.clone().neg()),
            AnyTensor::T2(x) => AnyTensor::T2(x.clone().neg()),
            AnyTensor::T3(x) => AnyTensor::T3(x.clone().neg()),
            AnyTensor::T4(x) => AnyTensor::T4(x.clone().neg()),
        },
        TensorInstr::Abs(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(x.clone().abs()),
            AnyTensor::T2(x) => AnyTensor::T2(x.clone().abs()),
            AnyTensor::T3(x) => AnyTensor::T3(x.clone().abs()),
            AnyTensor::T4(x) => AnyTensor::T4(x.clone().abs()),
        },
        TensorInstr::Exp(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(x.clone().exp()),
            AnyTensor::T2(x) => AnyTensor::T2(x.clone().exp()),
            AnyTensor::T3(x) => AnyTensor::T3(x.clone().exp()),
            AnyTensor::T4(x) => AnyTensor::T4(x.clone().exp()),
        },
        TensorInstr::Log(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(if cfg.safe_math { x.clone().clamp(1e-6,1e6).log() } else { x.clone().log() }),
            AnyTensor::T2(x) => AnyTensor::T2(if cfg.safe_math { x.clone().clamp(1e-6,1e6).log() } else { x.clone().log() }),
            AnyTensor::T3(x) => AnyTensor::T3(if cfg.safe_math { x.clone().clamp(1e-6,1e6).log() } else { x.clone().log() }),
            AnyTensor::T4(x) => AnyTensor::T4(if cfg.safe_math { x.clone().clamp(1e-6,1e6).log() } else { x.clone().log() }),
        },
        TensorInstr::Sqrt(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(if cfg.safe_math { x.clone().clamp(0.0,1e6).sqrt() } else { x.clone().sqrt() }),
            AnyTensor::T2(x) => AnyTensor::T2(if cfg.safe_math { x.clone().clamp(0.0,1e6).sqrt() } else { x.clone().sqrt() }),
            AnyTensor::T3(x) => AnyTensor::T3(if cfg.safe_math { x.clone().clamp(0.0,1e6).sqrt() } else { x.clone().sqrt() }),
            AnyTensor::T4(x) => AnyTensor::T4(if cfg.safe_math { x.clone().clamp(0.0,1e6).sqrt() } else { x.clone().sqrt() }),
        },

        // These may or may not exist in all backends; keep safe fallback.
        TensorInstr::Cos(r) => regs[idx(r,n)].passthrough(),
        TensorInstr::Sin(r) => regs[idx(r,n)].passthrough(),

        TensorInstr::Relu(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(activation::relu(x.clone())),
            AnyTensor::T2(x) => AnyTensor::T2(activation::relu(x.clone())),
            AnyTensor::T3(x) => AnyTensor::T3(activation::relu(x.clone())),
            AnyTensor::T4(x) => AnyTensor::T4(activation::relu(x.clone())),
        },
        TensorInstr::Sigmoid(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(activation::sigmoid(x.clone())),
            AnyTensor::T2(x) => AnyTensor::T2(activation::sigmoid(x.clone())),
            AnyTensor::T3(x) => AnyTensor::T3(activation::sigmoid(x.clone())),
            AnyTensor::T4(x) => AnyTensor::T4(activation::sigmoid(x.clone())),
        },
        TensorInstr::Tanh(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(activation::tanh(x.clone())),
            AnyTensor::T2(x) => AnyTensor::T2(activation::tanh(x.clone())),
            AnyTensor::T3(x) => AnyTensor::T3(activation::tanh(x.clone())),
            AnyTensor::T4(x) => AnyTensor::T4(activation::tanh(x.clone())),
        },

        // Keep reductions conservative for now: passthrough.
        TensorInstr::SumAll(r) => regs[idx(r,n)].passthrough(),
        TensorInstr::MeanAll(r) => regs[idx(r,n)].passthrough(),
        TensorInstr::SumDim(r, _d) => regs[idx(r,n)].passthrough(),
        TensorInstr::MeanDim(r, _d) => regs[idx(r,n)].passthrough(),
        TensorInstr::ArgMax(r, _d) => regs[idx(r,n)].passthrough(),

        // Rank-2 only ops (others passthrough)
        TensorInstr::Transpose(r) => match &regs[idx(r,n)] {
            AnyTensor::T2(x) => AnyTensor::T2(x.clone().transpose()),
            _ => regs[idx(r,n)].passthrough(),
        },
        TensorInstr::Matmul(a,b) => match (&regs[idx(a,n)], &regs[idx(b,n)]) {
            (AnyTensor::T2(x), AnyTensor::T2(y)) => AnyTensor::T2(x.clone().matmul(y.clone())),
            _ => regs[idx(a,n)].passthrough(),
        },
        TensorInstr::Concat(a,b,_d) => match (&regs[idx(a,n)], &regs[idx(b,n)]) {
            (AnyTensor::T2(x), AnyTensor::T2(y)) => AnyTensor::T2(Tensor::cat(vec![x.clone(), y.clone()], 0)),
            _ => regs[idx(a,n)].passthrough(),
        },
        TensorInstr::Repeat(r,_d,_c) => regs[idx(r,n)].passthrough(),
        TensorInstr::Slice(r,_d,_len) => regs[idx(r,n)].passthrough(),

        TensorInstr::Powf(r,_c) => regs[idx(r,n)].passthrough(),

        TensorInstr::Clamp(r) => match &regs[idx(r,n)] {
            AnyTensor::T1(x) => AnyTensor::T1(x.clone().clamp(-1e6,1e6)),
            AnyTensor::T2(x) => AnyTensor::T2(x.clone().clamp(-1e6,1e6)),
            AnyTensor::T3(x) => AnyTensor::T3(x.clone().clamp(-1e6,1e6)),
            AnyTensor::T4(x) => AnyTensor::T4(x.clone().clamp(-1e6,1e6)),
        },
    }
}