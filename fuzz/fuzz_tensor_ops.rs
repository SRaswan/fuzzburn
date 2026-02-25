//! Fuzz frontend tensor API (plain, no autograd)
//! env config: FUZZ_MODE (default panic)

#![no_main]

use fuzzburn::ir::{interpreter, TensorProgram, FuzzConfig, HarnessMode};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|prog: TensorProgram| {
    let cfg = FuzzConfig::from_env();
    if prog.ops.len() < cfg.min_ops { return; }

    if let Err(msg) = interpreter::run_tensor_program(&prog, &cfg) {
        let display = prog.ssa(&cfg);
        match cfg.mode {
            HarnessMode::PanicOnFirstError => panic!("fuzz_tensor_ops CRASH:\n{display}\nerror: {msg}"),
            HarnessMode::Continuous => eprintln!("\n=== CRASH (continuing) ===\n{display}\nerror: {msg}\n"),
        }
    }
});
