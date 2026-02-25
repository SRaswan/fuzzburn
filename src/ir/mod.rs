//! IR / AST module for fuzz programs.

pub mod ops;
pub mod shape;
pub mod program;
pub mod interpreter;
pub mod generate;

pub use program::{FuzzConfig, HarnessMode};
