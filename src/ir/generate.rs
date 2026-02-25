//! Shape-aware state-machine program generator.
//!
//! Instead of flat `#[derive(Arbitrary)]`, the generator maintains an arena of
//! live register shapes and at each step either:
//!
//!  1. **New leaf** (autograd only) — introduces a new `requires_grad` tensor.
//!     The leaf shape can be fully random or can *inherit* one dimension from
//!     an existing register.
//!
//!  2. **Operation** — examines the arena and picks a mathematically legal op.
//!     Unary / shape-changing ops are always legal.  Binary ops search for a
//!     compatible pair; if none exists they fall back to a guaranteed-legal unary.
//!
//! Both [`AutogradProgram`] and [`TensorProgram`] share the same `ProgramBuilder`
//! and op-generation logic.

use std::sync::OnceLock;

use arbitrary::{Arbitrary, Error as ArbError, Unstructured};

use super::ops::{BinaryKind, DiffOp, OpCategory, Reg, TensorInstr, UnaryKind};
use super::program::{AutogradProgram, TensorProgram};
use super::shape::Shape2;

/// Probability (0–100) that `pick_reg` / `pick_candidate` return the
/// most-recently-written register (the "sink") instead of a uniform draw.
/// Read once from `FUZZ_SINK_BIAS`; defaults to 60.
fn sink_bias() -> u8 {
    static VAL: OnceLock<u8> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("BIAS")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(60)
            .clamp(0, 100)
    })
}

/// Maximum distinct leaf tensors the builder will introduce (including r0).
const MAX_BUILDER_LEAVES: usize = 4;
/// Maximum total SSA steps (leaf introductions + operations).
const MAX_STEPS: usize = 48;

/// Cap on any single dimension to bound memory from Concat / Repeat.
/// NOTE: this only constrains *growth ops*; leaf dims follow FUZZ_MIN_DIM/FUZZ_MAX_DIM.
const MAX_GROW_DIM: usize = 48;

// ─── env helpers ─────────────────────────────────────────────────────────────

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Read FUZZ_MIN_DIM / FUZZ_MAX_DIM from the environment, with sane caps.
/// We cap to 255 because leaf dims are stored as u8 in the program IR.
fn fuzz_dim_bounds_u8() -> (u8, u8) {
    let min_dim = env_usize("FUZZ_MIN_DIM", 1).clamp(1, 4096);
    let max_dim = env_usize("FUZZ_MAX_DIM", 16).clamp(min_dim, 4096);

    let min_u8 = (min_dim.min(255)) as u8;
    let max_u8 = (max_dim.min(255)) as u8;

    // ensure min <= max (in case user sets FUZZ_MIN_DIM > 255)
    let min_u8 = min_u8.min(max_u8);
    (min_u8, max_u8)
}

// ─── builder state machine ──────────────────────────────────────────────────

struct ProgramBuilder {
    /// Shape of every register in the arena (index = register number).
    arena: Vec<Shape2>,
    /// Accumulated SSA ops.
    ops: Vec<DiffOp>,
    /// Pool of seed-byte vectors for leaf tensors.
    leaf_seeds: Vec<Vec<u8>>,
    /// How many leaves have been introduced so far (including r0).
    leaf_count: usize,
    /// r0 shape (stored separately so we can emit it in `AutogradProgram`).
    r0_rows: u8,
    r0_cols: u8,
}

impl ProgramBuilder {
    fn new() -> Self {
        ProgramBuilder {
            arena: Vec::with_capacity(MAX_STEPS + MAX_BUILDER_LEAVES),
            ops: Vec::with_capacity(MAX_STEPS),
            leaf_seeds: Vec::with_capacity(MAX_BUILDER_LEAVES),
            leaf_count: 0,
            r0_rows: 1,
            r0_cols: 1,
        }
    }

    /// Register the seed leaf (r0). Called exactly once before stepping.
    fn add_seed_leaf(&mut self, rows: u8, cols: u8, seed: Vec<u8>) {
        let (min_u8, max_u8) = fuzz_dim_bounds_u8();

        let rr = rows.clamp(min_u8, max_u8);
        let cc = cols.clamp(min_u8, max_u8);

        let r = rr as usize;
        let c = cc as usize;

        self.r0_rows = rr;
        self.r0_cols = cc;

        self.arena.push(Shape2(r, c));
        self.leaf_seeds.push(seed);
        self.leaf_count = 1;
    }

    /// One step of the state machine.
    fn step(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let can_leaf = self.leaf_count < MAX_BUILDER_LEAVES;
        let want_leaf: bool = can_leaf && u.arbitrary()?;

        if want_leaf {
            self.gen_leaf(u)
        } else {
            self.gen_operation(u)
        }
    }

    // ── leaf generation ─────────────────────────────────────────────────────

    fn gen_leaf(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        if self.leaf_count >= MAX_BUILDER_LEAVES {
            return self.gen_operation(u);
        }

        let (min_u8, max_u8) = fuzz_dim_bounds_u8();

        // 0 = fresh random
        // 1 = inherit rows   (same row-count as an existing register)
        // 2 = inherit cols   (same col-count as an existing register)
        // 3 = matmul-ready   (new.rows = existing.cols → enables left-matmul)
        let mode: u8 = u.int_in_range(0..=3)?;
        let dim = match mode {
            1 if !self.arena.is_empty() => {
                let (idx, _) = self.pick_reg(u)?;
                let src = self.arena[idx];
                Shape2(src.rows(), u.int_in_range(min_u8..=max_u8)? as usize)
            }
            2 if !self.arena.is_empty() => {
                let (idx, _) = self.pick_reg(u)?;
                let src = self.arena[idx];
                Shape2(u.int_in_range(min_u8..=max_u8)? as usize, src.cols())
            }
            3 if !self.arena.is_empty() => {
                let (idx, _) = self.pick_reg(u)?;
                let src = self.arena[idx];
                Shape2(src.cols(), u.int_in_range(min_u8..=max_u8)? as usize)
            }
            _ => Shape2(
                u.int_in_range(min_u8..=max_u8)? as usize,
                u.int_in_range(min_u8..=max_u8)? as usize,
            ),
        };

        let seed_len: usize = u.int_in_range(1..=16)?;
        let seed_data: Vec<u8> = (0..seed_len)
            .map(|_| u.arbitrary())
            .collect::<Result<_, _>>()?;

        let pool_idx = self.leaf_seeds.len();
        self.leaf_seeds.push(seed_data);

        self.ops.push(DiffOp::Leaf {
            seed: pool_idx as u8,
            rows: dim.rows().min(255) as u8,
            cols: dim.cols().min(255) as u8,
        });
        self.arena.push(dim);
        self.leaf_count += 1;
        Ok(())
    }

    // ── operation generation ────────────────────────────────────────────────

    fn gen_operation(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        debug_assert!(!self.arena.is_empty());

        let cat: u8 = u.int_in_range(0..=10)?;
        match cat {
            0..=2 => self.gen_unary(u),        // ~27 % unary element-wise
            3..=4 => self.gen_binary(u),       // ~18 % add / sub / mul
            5 => self.gen_matmul(u),           // ~ 9 %
            6 => self.gen_transpose(u),        // ~ 9 %
            7 => self.gen_dim_reduce(u),       // ~ 9 % sum_dim / mean_dim
            8 => self.gen_full_reduce(u),      // ~ 9 % sum_all / mean_all
            9 => self.gen_concat(u),           // ~ 9 %
            _ => self.gen_repeat(u),           // ~ 9 %
        }
    }

    /// Pick a register index biased toward the most recent (deepest) register.
    ///
    /// With probability `FUZZ_SINK_BIAS`% returns the last-written register
    /// (the current sink of the dataflow graph).  Otherwise picks uniformly.
    fn pick_reg(&self, u: &mut Unstructured) -> Result<(usize, Reg), ArbError> {
        let last = self.arena.len() - 1;
        let roll: u8 = u.int_in_range(0..=99)?;
        let idx = if roll < sink_bias() {
            last
        } else {
            u.int_in_range(0..=last)?
        };
        Ok((idx, Reg(idx as u8)))
    }

    /// Push a `TensorInstr` op and record its output shape.
    fn push_instr(&mut self, instr: TensorInstr, out: Shape2) {
        self.ops.push(DiffOp::Instr(instr));
        self.arena.push(out);
    }

    /// From a non-empty candidate list, pick an index biased toward the last
    /// (highest-index) candidate — same sink-bias coin flip as [`pick_reg`].
    fn pick_candidate(&self, cands: &[usize], u: &mut Unstructured) -> Result<usize, ArbError> {
        let last = cands.len() - 1;
        let roll: u8 = u.int_in_range(0..=99)?;
        let ci = if roll < sink_bias() {
            last
        } else {
            u.int_in_range(0..=last)?
        };
        Ok(cands[ci])
    }

    // ── individual op generators ────────────────────────────────────────────

    fn gen_unary(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (idx, reg) = self.pick_reg(u)?;
        let shape = self.arena[idx];
        let instr = match u.arbitrary::<UnaryKind>()? {
            UnaryKind::Neg     => TensorInstr::Neg(reg),
            UnaryKind::Abs     => TensorInstr::Abs(reg),
            UnaryKind::Exp     => TensorInstr::Exp(reg),
            UnaryKind::Log     => TensorInstr::Log(reg),
            UnaryKind::Sqrt    => TensorInstr::Sqrt(reg),
            UnaryKind::Cos     => TensorInstr::Cos(reg),
            UnaryKind::Sin     => TensorInstr::Sin(reg),
            UnaryKind::Relu    => TensorInstr::Relu(reg),
            UnaryKind::Sigmoid => TensorInstr::Sigmoid(reg),
            UnaryKind::Tanh    => TensorInstr::Tanh(reg),
            UnaryKind::Clamp   => TensorInstr::Clamp(reg),
        };
        self.push_instr(instr, shape);
        Ok(())
    }

    fn gen_transpose(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (idx, reg) = self.pick_reg(u)?;
        let d = self.arena[idx];
        self.push_instr(TensorInstr::Transpose(reg), Shape2(d.cols(), d.rows()));
        Ok(())
    }

    fn gen_full_reduce(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (_idx, reg) = self.pick_reg(u)?;
        let instr = if u.arbitrary()? {
            TensorInstr::MeanAll(reg)
        } else {
            TensorInstr::SumAll(reg)
        };
        self.push_instr(instr, Shape2(1, 1));
        Ok(())
    }

    fn gen_dim_reduce(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (idx, reg) = self.pick_reg(u)?;
        let s = self.arena[idx];
        let dim: u8 = if u.arbitrary()? { 1 } else { 0 };
        let out = if dim == 0 {
            Shape2(1, s.cols())
        } else {
            Shape2(s.rows(), 1)
        };
        let instr = match u.int_in_range(0..=2u8)? {
            0 => TensorInstr::SumDim(reg, dim),
            1 => TensorInstr::MeanDim(reg, dim),
            _ => TensorInstr::ArgMax(reg, dim),
        };
        self.push_instr(instr, out);
        Ok(())
    }

    fn gen_binary(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (ai, a_reg) = self.pick_reg(u)?;
        let sa = self.arena[ai];

        // Self is always broadcast-compatible, so cands is never empty.
        let cands: Vec<usize> = (0..self.arena.len())
            .filter(|&i| sa.broadcast_compatible(self.arena[i]))
            .collect();

        let bi = self.pick_candidate(&cands, u)?;
        let sb = self.arena[bi];
        let out = sa.broadcast_result(sb);
        let b_reg = Reg(bi as u8);

        let instr = match u.arbitrary::<BinaryKind>()? {
            BinaryKind::Add => TensorInstr::Add(a_reg, b_reg),
            BinaryKind::Sub => TensorInstr::Sub(a_reg, b_reg),
            BinaryKind::Mul => TensorInstr::Mul(a_reg, b_reg),
            BinaryKind::Div => TensorInstr::Div(a_reg, b_reg),
        };
        self.push_instr(instr, out);
        Ok(())
    }

    fn gen_matmul(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (ai, a_reg) = self.pick_reg(u)?;
        let sa = self.arena[ai];

        let cands: Vec<usize> = (0..self.arena.len())
            .filter(|&i| sa.cols() == self.arena[i].rows())
            .collect();

        if cands.is_empty() {
            // No matmul-compatible tensor exists — fall back to unary.
            return self.gen_unary(u);
        }

        let bi = self.pick_candidate(&cands, u)?;
        let sb = self.arena[bi];
        self.push_instr(
            TensorInstr::Matmul(a_reg, Reg(bi as u8)),
            Shape2(sa.rows(), sb.cols()),
        );
        Ok(())
    }

    fn gen_concat(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (ai, a_reg) = self.pick_reg(u)?;
        let sa = self.arena[ai];
        let dim: u8 = if u.arbitrary()? { 1 } else { 0 };

        let cands: Vec<usize> = (0..self.arena.len())
            .filter(|&i| {
                let sb = self.arena[i];
                if dim == 0 {
                    sa.cols() == sb.cols() && sa.rows() + sb.rows() <= MAX_GROW_DIM
                } else {
                    sa.rows() == sb.rows() && sa.cols() + sb.cols() <= MAX_GROW_DIM
                }
            })
            .collect();

        if cands.is_empty() {
            return self.gen_unary(u);
        }

        let bi = self.pick_candidate(&cands, u)?;
        let sb = self.arena[bi];
        let out = if dim == 0 {
            Shape2(sa.rows() + sb.rows(), sa.cols())
        } else {
            Shape2(sa.rows(), sa.cols() + sb.cols())
        };
        self.push_instr(TensorInstr::Concat(a_reg, Reg(bi as u8), dim), out);
        Ok(())
    }

    fn gen_repeat(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (idx, reg) = self.pick_reg(u)?;
        let s = self.arena[idx];
        let dim: u8 = if u.arbitrary()? { 1 } else { 0 };
        let cur = if dim == 0 { s.rows() } else { s.cols() };
        let max_times = (MAX_GROW_DIM / cur).min(4).max(1);
        let times: u8 = u.int_in_range(1..=max_times as u8)?;
        let out = if dim == 0 {
            Shape2(s.rows() * times as usize, s.cols())
        } else {
            Shape2(s.rows(), s.cols() * times as usize)
        };
        self.push_instr(TensorInstr::Repeat(reg, dim, times), out);
        Ok(())
    }
    fn gen_powf(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (idx, reg) = self.pick_reg(u)?;
        let shape = self.arena[idx];
        let c: u8 = u.int_in_range(0..=4)?;
        self.push_instr(TensorInstr::Powf(reg, c), shape);
        Ok(())
    }

    fn gen_slice(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let (idx, reg) = self.pick_reg(u)?;
        let s = self.arena[idx];
        let dim: u8 = if u.arbitrary()? { 1 } else { 0 };
        let size = if dim == 0 { s.rows() } else { s.cols() };
        if size <= 1 {
            return self.gen_unary(u);
        }
        let take: u8 = u.int_in_range(1..=size as u8 - 1)?;
        let out = if dim == 0 {
            Shape2(take as usize, s.cols())
        } else {
            Shape2(s.rows(), take as usize)
        };
        self.push_instr(TensorInstr::Slice(reg, dim, take), out);
        Ok(())
    }
    // ── finalise ────────────────────────────────────────────────────────────

    fn build_autograd(self) -> AutogradProgram {
        AutogradProgram {
            rows: self.r0_rows,
            cols: self.r0_cols,
            leaf_seeds: self.leaf_seeds,
            ops: self.ops,
            shapes: self.arena,
        }
    }

    fn build_tensor(self) -> TensorProgram {
        let instrs: Vec<TensorInstr> = self.ops.into_iter().filter_map(|op| {
            match op {
                DiffOp::Instr(i) => Some(i),
                _ => None,
            }
        }).collect();
        // shapes[0] = r0, shapes[1..] from ops.  Since we filtered leaves
        // that never appear in a TensorProgram build, the arena is already
        // just r0 + one shape per TensorInstr.
        TensorProgram {
            rows: self.r0_rows,
            cols: self.r0_cols,
            values: self.leaf_seeds.into_iter().next().unwrap_or_default(),
            ops: instrs,
            shapes: self.arena,
        }
    }
}

// ─── Arbitrary impl ─────────────────────────────────────────────────────────

impl<'a> Arbitrary<'a> for AutogradProgram {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self, ArbError> {
        let mut b = ProgramBuilder::new();

        let (min_u8, max_u8) = fuzz_dim_bounds_u8();

        // r0: seed leaf (always present)
        let rows: u8 = u.int_in_range(min_u8..=max_u8)?;
        let cols: u8 = u.int_in_range(min_u8..=max_u8)?;
        let seed_len: usize = u.int_in_range(1..=16)?;
        let seed: Vec<u8> = (0..seed_len).map(|_| u.arbitrary()).collect::<Result<_, _>>()?;
        b.add_seed_leaf(rows, cols, seed);

        // Generate program steps until fuzzer bytes are exhausted or limit hit
        let num_steps: usize = u.int_in_range(1..=MAX_STEPS)?;
        for _ in 0..num_steps {
            if u.is_empty() {
                break;
            }
            b.step(u)?;
        }

        Ok(b.build_autograd())
    }
}

// ─── Arbitrary impl for TensorProgram ───────────────────────────────────────

impl<'a> Arbitrary<'a> for TensorProgram {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self, ArbError> {
        let mut b = ProgramBuilder::new();

        // r0: single input tensor
        let rows: u8 = u.int_in_range(1..=16)?;
        let cols: u8 = u.int_in_range(1..=16)?;
        let seed_len: usize = u.int_in_range(1..=16)?;
        let seed: Vec<u8> = (0..seed_len)
            .map(|_| u.arbitrary())
            .collect::<Result<_, _>>()?;
        b.add_seed_leaf(rows, cols, seed);

        // Only operations, no extra leaves
        let num_steps: usize = u.int_in_range(1..=MAX_STEPS)?;
        for _ in 0..num_steps {
            if u.is_empty() {
                break;
            }
            b.gen_operation(u)?;
        }

        Ok(b.build_tensor())
    }
}
