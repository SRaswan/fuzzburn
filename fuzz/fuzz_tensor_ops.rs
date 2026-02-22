//! Fuzz frontend tensor API (plain, no autograd)
//! env config: FUZZ_MODE (default panic)

#![no_main]

use fuzzburn::ir::{interpreter, FuzzConfig};
use fuzzburn::ir::program::TensorProgram;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|prog: TensorProgram| {
    let config = FuzzConfig::from_env();
    interpreter::run_tensor_program(&prog, config.mode);
});
