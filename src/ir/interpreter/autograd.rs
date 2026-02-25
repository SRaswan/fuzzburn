use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use crate::ir::ops::DiffOp;
use crate::ir::program::{AutogradProgram, FuzzConfig};
use crate::ir::shape::ShapeN;

use super::{AnyTensor, bytes_to_floats, catch_as_result, eval_tensor_instr_direct};

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

type DiffB = Autodiff<NdArray>;
#[cfg(feature = "oracle-tch")]
type DiffTch = Autodiff<LibTorch>;

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

fn make_leaf_any<AB: AutodiffBackend>(raw: &[u8], shape: ShapeN, device: &AB::Device) -> AnyTensor<AB> {
    let data = bytes_to_floats(raw, shape.numel());
    match shape.rank {
        1 => AnyTensor::T1(
            Tensor::<AB,1>::from_floats(data.as_slice(), device)
                .reshape([shape.dims[0]])
                .require_grad()
        ),
        2 => AnyTensor::T2(
            Tensor::<AB,1>::from_floats(data.as_slice(), device)
                .reshape([shape.dims[0], shape.dims[1]])
                .require_grad()
        ),
        3 => AnyTensor::T3(
            Tensor::<AB,1>::from_floats(data.as_slice(), device)
                .reshape([shape.dims[0], shape.dims[1], shape.dims[2]])
                .require_grad()
        ),
        4 => AnyTensor::T4(
            Tensor::<AB,1>::from_floats(data.as_slice(), device)
                .reshape([shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]])
                .require_grad()
        ),
        _ => unreachable!(),
    }
}

fn backward_any<AB: AutodiffBackend>(t: &AnyTensor<AB>) -> <AB as AutodiffBackend>::Gradients {
    match t {
        AnyTensor::T1(x) => x.clone().backward(),
        AnyTensor::T2(x) => x.clone().backward(),
        AnyTensor::T3(x) => x.clone().backward(),
        AnyTensor::T4(x) => x.clone().backward(),
    }
}

fn grad_to_vec_or_zeros<AB: AutodiffBackend>(
    leaf: &AnyTensor<AB>,
    grads: &<AB as AutodiffBackend>::Gradients,
    numel: usize,
) -> Vec<f32> {
    match leaf {
        AnyTensor::T1(x) => x.grad(grads).map(|g| g.into_data().to_vec::<f32>().unwrap_or_default()).unwrap_or(vec![0.0; numel]),
        AnyTensor::T2(x) => x.grad(grads).map(|g| g.into_data().to_vec::<f32>().unwrap_or_default()).unwrap_or(vec![0.0; numel]),
        AnyTensor::T3(x) => x.grad(grads).map(|g| g.into_data().to_vec::<f32>().unwrap_or_default()).unwrap_or(vec![0.0; numel]),
        AnyTensor::T4(x) => x.grad(grads).map(|g| g.into_data().to_vec::<f32>().unwrap_or_default()).unwrap_or(vec![0.0; numel]),
    }
}

fn collect_grads<AB: AutodiffBackend>(prog: &AutogradProgram, cfg: &FuzzConfig, device: &AB::Device) -> Vec<Vec<f32>> {
    let r0_shape = prog.shapes.get(0).copied().unwrap_or_else(|| {
        let r = (prog.rows as usize).clamp(cfg.min_dim, cfg.max_dim).max(1);
        let c = (prog.cols as usize).clamp(cfg.min_dim, cfg.max_dim).max(1);
        ShapeN::new(2, [r,c,1,1])
    });
    let r0_shape = clamp_shape_to_budget(r0_shape, cfg);

    let leaf0_raw = prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]);
    let r0 = make_leaf_any::<AB>(leaf0_raw, r0_shape, device);

    let mut regs: Vec<AnyTensor<AB>> = vec![r0];
    let mut leaf_indices: Vec<usize> = vec![0];
    let mut leaf_shapes: Vec<ShapeN> = vec![r0_shape];
    let mut leaf_count = 1usize;

    for (i, op) in prog.ops.iter().enumerate() {
        let out_shape = prog.shapes.get(i + 1).copied().unwrap_or(r0_shape);
        let out_shape = clamp_shape_to_budget(out_shape, cfg);

        let val = match op {
            DiffOp::Leaf { seed, .. } => {
                if leaf_count < cfg.max_leaves {
                    let pool_idx = if prog.leaf_seeds.is_empty() { 0 } else { (*seed as usize) % prog.leaf_seeds.len() };
                    let raw = prog.leaf_seeds.get(pool_idx).map(Vec::as_slice).unwrap_or(&[]);
                    let leaf = make_leaf_any::<AB>(raw, out_shape, device);
                    leaf_indices.push(regs.len());
                    leaf_shapes.push(out_shape);
                    leaf_count += 1;
                    leaf
                } else {
                    // alias existing
                    let alias = (*seed as usize) % regs.len();
                    regs[alias].clone()
                }
            }
            DiffOp::Instr(instr) => {
                eval_tensor_instr_direct(&regs, instr, cfg)
            }
        };
        regs.push(val);
    }

    let last = regs.last().expect("no regs").clone();
    let grads = backward_any::<AB>(&last);

    leaf_indices.iter().zip(leaf_shapes.iter())
        .map(|(&ri, sh)| grad_to_vec_or_zeros::<AB>(&regs[ri], &grads, sh.numel()))
        .collect()
}

pub fn run_autograd_program(prog: &AutogradProgram, cfg: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = collect_grads::<DiffB>(prog, cfg, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        {
            let lt = collect_grads::<DiffTch>(prog, cfg, &LibTorchDevice::Cpu);
            assert_eq!(nd.len(), lt.len(), "oracle leaf count mismatch");
            for (i, (a, b)) in nd.iter().zip(lt.iter()).enumerate() {
                compare_outputs(a, b, &format!("grad leaf #{i}"), cfg);
            }
        }
    }))
}