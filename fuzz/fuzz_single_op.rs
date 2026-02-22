//! Fuzz single Burn tensor operation 
//! env config: FUZZ_MODE (default panic)
#![no_main]

use fuzzburn::ir::{interpreter, SingleOpCase, FuzzConfig};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|case: SingleOpCase| {
    let config = FuzzConfig::from_env();
    interpreter::run_single_op_case(&case, config.mode);
});
