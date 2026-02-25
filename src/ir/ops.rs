//! SSA instructions for fuzz IR.

use std::fmt;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug, Clone, Copy)]
pub struct Reg(pub u8);

impl Reg {
    pub fn name(self) -> String { format!("r{}", self.0) }
}
impl From<Reg> for usize {
    fn from(r: Reg) -> usize { r.0 as usize }
}
impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "r{}", self.0) }
}

#[derive(Arbitrary, Debug, Clone, Copy)]
pub enum UnaryKind {
    Neg, Abs, Exp, Log, Sqrt, Relu, Sigmoid, Tanh, Clamp,
    // present in enum list but interpreter may fallback if unsupported:
    Cos, Sin,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
pub enum BinaryKind {
    Add, Sub, Mul, Div,
}

#[derive(Arbitrary, Debug, Clone)]
pub enum TensorInstr {
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),

    Neg(Reg),
    Abs(Reg),
    Exp(Reg),
    Log(Reg),
    Sqrt(Reg),
    Cos(Reg),
    Sin(Reg),

    Relu(Reg),
    Sigmoid(Reg),
    Tanh(Reg),

    // reductions (may fallback for rank != 2)
    SumAll(Reg),
    MeanAll(Reg),
    SumDim(Reg, u8),
    MeanDim(Reg, u8),
    ArgMax(Reg, u8),

    Transpose(Reg),
    Concat(Reg, Reg, u8),
    Repeat(Reg, u8, u8),
    Slice(Reg, u8, u8),

    Matmul(Reg, Reg),

    Powf(Reg, u8),

    Clamp(Reg),
}

impl TensorInstr {
    pub fn ssa_line(&self, out: &str) -> String {
        match self {
            TensorInstr::Add(a,b) => format!("{out} = {} + {}", a.name(), b.name()),
            TensorInstr::Sub(a,b) => format!("{out} = {} - {}", a.name(), b.name()),
            TensorInstr::Mul(a,b) => format!("{out} = {} * {}", a.name(), b.name()),
            TensorInstr::Div(a,b) => format!("{out} = {} / {}", a.name(), b.name()),

            TensorInstr::Neg(r) => format!("{out} = -{}", r.name()),
            TensorInstr::Abs(r) => format!("{out} = abs({})", r.name()),
            TensorInstr::Exp(r) => format!("{out} = exp({})", r.name()),
            TensorInstr::Log(r) => format!("{out} = log({})", r.name()),
            TensorInstr::Sqrt(r)=> format!("{out} = sqrt({})", r.name()),
            TensorInstr::Cos(r) => format!("{out} = cos({})", r.name()),
            TensorInstr::Sin(r) => format!("{out} = sin({})", r.name()),

            TensorInstr::Relu(r)    => format!("{out} = relu({})", r.name()),
            TensorInstr::Sigmoid(r) => format!("{out} = sigmoid({})", r.name()),
            TensorInstr::Tanh(r)    => format!("{out} = tanh({})", r.name()),

            TensorInstr::SumAll(r)  => format!("{out} = sum({})", r.name()),
            TensorInstr::MeanAll(r) => format!("{out} = mean({})", r.name()),
            TensorInstr::SumDim(r,d)=> format!("{out} = sum({}, dim={})", r.name(), (*d as usize)%4),
            TensorInstr::MeanDim(r,d)=> format!("{out} = mean({}, dim={})", r.name(), (*d as usize)%4),
            TensorInstr::ArgMax(r,d)=> format!("{out} = argmax({}, dim={})", r.name(), (*d as usize)%4),

            TensorInstr::Transpose(r)=> format!("{out} = {}.T", r.name()),
            TensorInstr::Concat(a,b,d)=> format!("{out} = cat([{},{}], dim={})", a.name(), b.name(), (*d as usize)%4),
            TensorInstr::Repeat(r,d,c)=> format!("{out} = {}.repeat(dim={}, ×{})", r.name(), (*d as usize)%4, (*c as usize).clamp(1,4)),
            TensorInstr::Slice(r,d,len)=> format!("{out} = {}.slice(dim={}, 0..{})", r.name(), (*d as usize)%4, *len),

            TensorInstr::Matmul(a,b)=> format!("{out} = {} @ {}", a.name(), b.name()),
            TensorInstr::Powf(r, _c)=> format!("{out} = {}.powf(<exp>)", r.name()),

            TensorInstr::Clamp(r)=> format!("{out} = clamp({}, -1e6, 1e6)", r.name()),
        }
    }
}

#[derive(Arbitrary, Debug, Clone)]
pub enum DiffOp {
    Leaf { seed: u8, rank: u8, dims: [u16; 4] },
    Instr(TensorInstr),
}

impl DiffOp {
    pub fn ssa_line(&self, out: &str) -> String {
        match self {
            DiffOp::Leaf { rank, dims, .. } => {
                let r = (*rank).clamp(1,4) as usize;
                let mut parts = Vec::new();
                for i in 0..r {
                    parts.push((dims[i] as usize).max(1).to_string());
                }
                format!("{out} = leaf({})  [requires_grad]", parts.join("×"))
            }
            DiffOp::Instr(i) => i.ssa_line(out),
        }
    }
}