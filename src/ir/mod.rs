//! IR / AST module for fuzz programs.

pub mod interpreter;
pub mod ops;
pub mod program;

// Convenience re-exports so fuzz targets can write `fuzzburn::ir::FuzzConfig`.
pub use program::{FuzzConfig, HarnessMode, SingleOpCase};
