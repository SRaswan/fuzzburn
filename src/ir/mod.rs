//! IR / AST module for fuzz programs.

pub mod ops;
pub mod program;

pub use program::{FuzzConfig, HarnessMode};
pub mod interpreter;
