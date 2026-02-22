//! Fuzz the Burn frontend tensor API (plain, no autograd).
//!
//! The fuzz input is a [`TensorProgram`]: a sequence of [`TensorOp`]s applied
//! to a single seeded input tensor.  The interpreter walks the sequence and
//! dispatches to Burn's NdArray backend; any panic or abort is a bug.
//!
//! This target exercises multi-op paths: fusion decisions, intermediate
//! tensor allocation, lazy evaluation, and shape-tracking across ops.
//!
//! For single-op differential testing against PyTorch, see `fuzz_single_op`.
//!
//! # Environment variables
//!
//! | Variable    | Values                | Default  |
//! |-------------|-----------------------|----------|
//! | `FUZZ_MODE` | `panic` / `continuous`| `panic`  |
#![no_main]

use fuzzburn::ir::{interpreter, FuzzConfig};
use fuzzburn::ir::program::TensorProgram;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|prog: TensorProgram| {
    let config = FuzzConfig::from_env();
    interpreter::run_tensor_program(&prog, config.mode);
});
