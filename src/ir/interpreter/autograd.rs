//! Autograd program SSA interpreter with gradient collection.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use crate::ir::shape::Shape2;
use crate::ir::ops::DiffOp;
use crate::ir::program::{AutogradProgram, FuzzConfig};

use super::{bytes_to_floats, catch_as_result, eval_tensor_instr_direct};

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

type DiffB = Autodiff<NdArray>;

#[cfg(feature = "oracle-tch")]
type DiffTch = Autodiff<LibTorch>;

fn make_leaf<AB: AutodiffBackend>(
    raw: &[u8],
    rows: usize,
    cols: usize,
    device: &AB::Device,
) -> Tensor<AB, 2> {
    Tensor::<AB, 1>::from_floats(bytes_to_floats(raw, rows * cols).as_slice(), device)
        .reshape([rows, cols])
        .require_grad()
}

fn collect_grads<AB: AutodiffBackend>(
    prog: &AutogradProgram,
    config: &FuzzConfig,
    device: &AB::Device,
) -> Vec<Vec<f32>> {
    // r0 shape
    let (r0_rows, r0_cols) = prog.shapes.get(0).map(|s| (s.rows(), s.cols())).unwrap_or_else(|| {
        (
            (prog.rows as usize).clamp(config.min_dim, config.max_dim),
            (prog.cols as usize).clamp(config.min_dim, config.max_dim),
        )
    });

    let leaf0_raw = prog.leaf_seeds.first().map(Vec::as_slice).unwrap_or(&[]);
    let leaf_0 = make_leaf::<AB>(leaf0_raw, r0_rows, r0_cols, device);

    let mut regs: Vec<Tensor<AB, 2>> = vec![leaf_0];
    let mut leaf_indices: Vec<usize> = vec![0];
    let mut leaf_count: usize = 1;

    for (i, op) in prog.ops.iter().enumerate() {
        let val: Tensor<AB, 2> = match op {
            DiffOp::Leaf { seed, .. } => {
                // leaf register is r{i+1}, shape is shapes[i+1]
                let sh = prog.shapes.get(i + 1).copied().unwrap_or(Shape2(1, 1));

                if leaf_count < config.max_leaves {
                    let pool_idx = if prog.leaf_seeds.is_empty() {
                        0
                    } else {
                        (*seed as usize) % prog.leaf_seeds.len()
                    };
                    let raw = prog.leaf_seeds.get(pool_idx).map(Vec::as_slice).unwrap_or(&[]);

                    let leaf = make_leaf::<AB>(raw, sh.rows(), sh.cols(), device);
                    leaf_indices.push(regs.len());
                    leaf_count += 1;
                    leaf
                } else {
                    // alias an existing reg
                    let alias_idx = (*seed as usize) % regs.len();
                    regs[alias_idx].clone()
                }
            }
            DiffOp::Instr(instr) => {
                // generator made operands shape-legal
                eval_tensor_instr_direct(&regs, instr, config)
            }
        };

        regs.push(val);
    }

    // backward from last reg
    let last = regs.last().expect("register file is empty").clone();
    let grads = last.backward();

    leaf_indices
        .iter()
        .map(|&ri| {
            let sh = prog.shapes.get(ri).copied().unwrap_or(Shape2(1, 1));
            match regs[ri].grad(&grads) {
                Some(g) => g
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_else(|e| panic!("into_data for r{ri} grad failed: {e}")),
                None => vec![0.0_f32; sh.rows() * sh.cols()],
            }
        })
        .collect()
}

pub fn run_autograd_program(prog: &AutogradProgram, config: &FuzzConfig) -> Result<(), String> {
    catch_as_result(std::panic::AssertUnwindSafe(|| {
        let nd = collect_grads::<DiffB>(prog, config, &NdArrayDevice::default());
        #[cfg(feature = "oracle-tch")]
        run_autograd_oracle(prog, config, nd);
    }))
}

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