//! Fuzz a single Burn tensor operation applied to two independent input tensors.
//!
//! Each fuzz input is a [`SingleOpCase`]: one [`TensorOp`] applied to
//! independently seeded `lhs` and `rhs` tensors of shape `[rows, cols]`.
//! Binary ops (`Add`, `Sub`, `Mul`) receive genuinely distinct data on both
//! sides, unlike `fuzz_tensor_ops` where binary ops fold `t ⊕ t`.
//! Unary ops and reductions use only `lhs`; `rhs` is generated but ignored.
//!
//! Any panic or abort inside Burn is recorded as a crash by libFuzzer.
//!
//! # Oracle mode
//!
//! Compile with `--features oracle-tch` to also run the same op on the
//! LibTorch (PyTorch) backend and compare outputs element-wise.  This turns
//! the fuzzer into a differential tester: any numerical divergence between
//! Burn/NdArray and LibTorch is reported as a bug.
//!
//! Requires LibTorch to be installed; set the `LIBTORCH` environment variable
//! or follow the burn-tch setup guide before building with this feature.
//!
//! # Environment variables
//!
//! | Variable    | Values                | Default  |
//! |-------------|-----------------------|----------|
//! | `FUZZ_MODE` | `panic` / `continuous`| `panic`  |
#![no_main]

use fuzzburn::ir::{interpreter, SingleOpCase, FuzzConfig};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|case: SingleOpCase| {
    let config = FuzzConfig::from_env();
    interpreter::run_single_op_case(&case, config.mode);
});
