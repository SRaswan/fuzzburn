//! IR / AST module for fuzz programs.

pub mod interpreter;
pub mod ops;
pub mod program;

pub use program::{FuzzConfig, HarnessMode};
