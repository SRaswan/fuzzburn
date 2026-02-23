//! Plain tensor-program SSA interpreter.

use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[cfg(feature = "oracle-tch")]
use burn::backend::LibTorch;
#[cfg(feature = "oracle-tch")]
use burn::backend::libtorch::LibTorchDevice;

use crate::ir::shape::Shape2;
use super::{PlainB, bytes_to_floats, catch_as_result, eval_tensor_instr_direct};
use crate::ir::program::TensorProgram;

#[cfg(feature = "oracle-tch")]
use super::compare_outputs;

/// Run a plain SSA TensorProgram on any backend, returning the final
/// register's data as `Vec<f32>`.
fn eval_tensor_program<B: Backend>(prog: &TensorProgram, device: &B::Device) -> Vec<f32> {
    let s0 = prog.shapes.first().copied().unwrap_or(Shape2(1, 1));

    let r0: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(
            bytes_to_floats(&prog.values, s0.rows() * s0.cols()).as_slice(),
            device,
        )
        .reshape([s0.rows(), s0.cols()]);

    let mut regs: Vec<Tensor<B, 2>> = vec![r0];

    for instr in &prog.ops {
        let val = eval_tensor_instr_direct(&regs, instr);
        regs.push(val);
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

#[cfg(feature = "oracle-tch")]
fn run_tensor_program_oracle(prog: &TensorProgram, nd: Vec<f32>) {
    let lt = eval_tensor_program::<LibTorch>(prog, &LibTorchDevice::Cpu);
    compare_outputs(&nd, &lt, "tensor_program");
}
