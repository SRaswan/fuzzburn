use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::backend::Backend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use crate::ir::program::{FuzzConfig, TensorProgram};
use crate::ir::shape::ShapeN;

use super::{AnyTensor, PlainB, bytes_to_floats, catch_as_result, eval_tensor_instr_direct};

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

fn clamp_shape_to_budget(mut sh: ShapeN, cfg: &FuzzConfig) -> ShapeN {
    for i in 0..(sh.rank as usize) {
        sh.dims[i] = sh.dims[i].clamp(cfg.min_dim, cfg.max_dim).max(1);
    }
    while sh.numel() < cfg.min_array_elems {
        let last = (sh.rank as usize) - 1;
        sh.dims[last] = (sh.dims[last] * 2).min(cfg.max_dim).max(1);
        if sh.dims[last] == cfg.max_dim { break; }
    }
    while sh.numel() > cfg.max_array_elems {
        let last = (sh.rank as usize) - 1;
        sh.dims[last] = (sh.dims[last] / 2).max(1);
        if sh.dims[last] == 1 { break; }
    }
    sh
}

fn make_seed_any<B: Backend>(raw: &[u8], shape: ShapeN, device: &B::Device) -> AnyTensor<B> {
    let data = bytes_to_floats(raw, shape.numel());
    match shape.rank {
        1 => AnyTensor::T1(burn::tensor::Tensor::<B,1>::from_floats(data.as_slice(), device)
            .reshape([shape.dims[0]])),
        2 => AnyTensor::T2(burn::tensor::Tensor::<B,1>::from_floats(data.as_slice(), device)
            .reshape([shape.dims[0], shape.dims[1]])),
        3 => AnyTensor::T3(burn::tensor::Tensor::<B,1>::from_floats(data.as_slice(), device)
            .reshape([shape.dims[0], shape.dims[1], shape.dims[2]])),
        4 => AnyTensor::T4(burn::tensor::Tensor::<B,1>::from_floats(data.as_slice(), device)
            .reshape([shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]])),
        _ => unreachable!(),
    }
}

fn eval_tensor_program<B: Backend>(prog: &TensorProgram, device: &B::Device, cfg: &FuzzConfig) -> Vec<f32> {
    let r0_shape = prog.shapes.get(0).copied().unwrap_or_else(|| {
        let r = (prog.rows as usize).clamp(cfg.min_dim, cfg.max_dim).max(1);
        let c = (prog.cols as usize).clamp(cfg.min_dim, cfg.max_dim).max(1);
        ShapeN::new(2, [r,c,1,1])
    });
    let r0_shape = clamp_shape_to_budget(r0_shape, cfg);

    let r0 = make_seed_any::<B>(&prog.values, r0_shape, device);
    let mut regs: Vec<AnyTensor<B>> = vec![r0];

    for instr in &prog.ops {
        let val = eval_tensor_instr_direct(&regs, instr, cfg);
        regs.push(val);
    }

    regs.pop().expect("register file empty").to_vec_f32()
}

pub fn run_tensor_program(prog: &TensorProgram, cfg: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = eval_tensor_program::<PlainB>(prog, &NdArrayDevice::default(), cfg);
        #[cfg(feature = "oracle-tch")]
        {
            let lt = eval_tensor_program::<LibTorch>(prog, &LibTorchDevice::Cpu, cfg);
            compare_outputs(&nd, &lt, "tensor_program", cfg);
        }
    }))
}