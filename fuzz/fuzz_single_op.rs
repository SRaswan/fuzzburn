//! Fuzz single Burn tensor operation 
//! env config: FUZZ_MODE (default panic)
#![no_main]

use fuzzburn::ir::{interpreter, SingleOpCase, FuzzConfig, HarnessMode};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|case: SingleOpCase| {
    let config = FuzzConfig::from_env();
    if let Err(msg) = interpreter::run_single_op_case(&case) {
        match config.mode {
            HarnessMode::PanicOnFirstError => {
                panic!("fuzz_single_op CRASH:\n{case}\nerror: {msg}");
            }
            HarnessMode::Continuous => {
                eprintln!("\n=== CRASH (fuzz_single_op, continuing) ===\n{case}\nerror: {msg}\n");
            }
        }
    }
});
