//! Burn autograd fuzz


#![no_main]

use fuzzburn::ir::{interpreter, program::AutogradProgram, FuzzConfig};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|prog: AutogradProgram| {
    let config = FuzzConfig::from_env();
    println!(
        "=== Fuzz input (max_leaves={}) ===\n{}",
        config.max_leaves,
        prog.ssa(&config)
    );
    interpreter::run_autograd_program(&prog, &config);
});
