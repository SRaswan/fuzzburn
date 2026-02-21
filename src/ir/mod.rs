//! IR / AST module for fuzz programs.
//!
//! Hierarchy:
//!
//! ```text
//! ir/
//!   ops.rs          – instruction-set enums (TensorOp, DiffOp, TensorRef)
//!   program.rs      – AST root types (TensorProgram, AutogradProgram)
//!   interpreter.rs  – tree-walker that maps ops → Burn tensor calls
//! ```

pub mod interpreter;
pub mod ops;
pub mod program;
