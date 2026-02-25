use arbitrary::{Arbitrary, Error as ArbError, Unstructured};
use super::ops::{BinaryKind, DiffOp, Reg, TensorInstr, UnaryKind};
use super::program::{AutogradProgram, FuzzConfig, TensorProgram};
use super::shape::ShapeN;

fn read_cfg() -> FuzzConfig {
    // NOTE: generator can't accept &FuzzConfig from arbitrary, so read env here.
    FuzzConfig::from_env()
}

fn clamp_shape_to_budget(mut sh: ShapeN, cfg: &FuzzConfig) -> ShapeN {
    // clamp per dim
    for i in 0..(sh.rank as usize) {
        sh.dims[i] = sh.dims[i].clamp(cfg.min_dim, cfg.max_dim).max(1);
    }
    // enforce min/max element budget by growing/shrinking last dim
    while sh.numel() < cfg.min_array_elems {
        let last = (sh.rank as usize) - 1;
        sh.dims[last] = (sh.dims[last] * 2).min(cfg.max_dim).max(1);
        if sh.dims[last] == cfg.max_dim { break; }
    }
    while sh.numel() > cfg.max_array_elems {
        let last = (sh.rank as usize) - 1;
        sh.dims[last] = (sh.dims[last] / 2).max(1);
        if sh.dims[last] == 1 { break; }
    }
    sh
}

fn sample_shape_1to4(u: &mut Unstructured, cfg: &FuzzConfig) -> Result<ShapeN, ArbError> {
    let rank: u8 = u.int_in_range(1..=4)?;
    let mut dims = [1usize; 4];
    for i in 0..(rank as usize) {
        dims[i] = u.int_in_range(cfg.min_dim..=cfg.max_dim)?.max(1);
    }
    Ok(clamp_shape_to_budget(ShapeN::new(rank, dims), cfg))
}

fn seed_bytes(u: &mut Unstructured) -> Result<Vec<u8>, ArbError> {
    let len: usize = u.int_in_range(1..=16)?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(u.arbitrary::<u8>()?);
    }
    Ok(v)
}

struct Builder {
    cfg: FuzzConfig,
    arena: Vec<ShapeN>,
    ops: Vec<DiffOp>,
    leaf_seeds: Vec<Vec<u8>>,
    leaf_count: usize,
}

impl Builder {
    fn new(cfg: FuzzConfig) -> Self {
        Self {
            cfg,
            arena: Vec::new(),
            ops: Vec::new(),
            leaf_seeds: Vec::new(),
            leaf_count: 0,
        }
    }

    fn add_seed_leaf(&mut self, sh: ShapeN, seed: Vec<u8>) {
        self.arena.push(sh);
        self.leaf_seeds.push(seed);
        self.leaf_count = 1;
    }

    fn pick_reg(&self, u: &mut Unstructured) -> Result<usize, ArbError> {
        let n = self.arena.len();
        if n == 0 { return Ok(0); }
        let last = n - 1;
        let roll: u8 = u.int_in_range(0..=99)?;
        let idx = if roll < self.cfg.sink_bias {
            last
        } else {
            u.int_in_range(0..=last)?
        };
        Ok(idx)
    }

    fn push_instr(&mut self, instr: TensorInstr, out: ShapeN) {
        self.ops.push(DiffOp::Instr(instr));
        self.arena.push(out);
    }

    fn gen_leaf(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        if self.leaf_count >= self.cfg.max_leaves {
            return Ok(());
        }
        let sh = sample_shape_1to4(u, &self.cfg)?;
        let seed = seed_bytes(u)?;
        let pool_idx = self.leaf_seeds.len();
        self.leaf_seeds.push(seed);

        // store dims as u16 for IR
        let mut dims16 = [1u16; 4];
        for i in 0..(sh.rank as usize) {
            dims16[i] = sh.dims[i].min(u16::MAX as usize) as u16;
        }

        self.ops.push(DiffOp::Leaf {
            seed: pool_idx.min(255) as u8,
            rank: sh.rank,
            dims: dims16,
        });
        self.arena.push(sh);
        self.leaf_count += 1;
        Ok(())
    }

    fn gen_unary(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let idx = self.pick_reg(u)?;
        let sh = self.arena[idx];
        let r = Reg(idx as u8);
        let kind: UnaryKind = u.arbitrary()?;

        let instr = match kind {
            UnaryKind::Neg => TensorInstr::Neg(r),
            UnaryKind::Abs => TensorInstr::Abs(r),
            UnaryKind::Exp => TensorInstr::Exp(r),
            UnaryKind::Log => TensorInstr::Log(r),
            UnaryKind::Sqrt => TensorInstr::Sqrt(r),
            UnaryKind::Relu => TensorInstr::Relu(r),
            UnaryKind::Sigmoid => TensorInstr::Sigmoid(r),
            UnaryKind::Tanh => TensorInstr::Tanh(r),
            UnaryKind::Clamp => TensorInstr::Clamp(r),
            UnaryKind::Cos => TensorInstr::Cos(r),
            UnaryKind::Sin => TensorInstr::Sin(r),
        };
        self.push_instr(instr, sh);
        Ok(())
    }

    fn gen_binary(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        let ai = self.pick_reg(u)?;
        let sa = self.arena[ai];

        // pick a reg with SAME shape (keep it simple & legal across ranks)
        let mut cands: Vec<usize> = Vec::new();
        for (i, sh) in self.arena.iter().enumerate() {
            if *sh == sa { cands.push(i); }
        }
        let bi = cands[u.int_in_range(0..=cands.len()-1)?];
        let a = Reg(ai as u8);
        let b = Reg(bi as u8);

        let kind: BinaryKind = u.arbitrary()?;
        let instr = match kind {
            BinaryKind::Add => TensorInstr::Add(a,b),
            BinaryKind::Sub => TensorInstr::Sub(a,b),
            BinaryKind::Mul => TensorInstr::Mul(a,b),
            BinaryKind::Div => TensorInstr::Div(a,b),
        };
        self.push_instr(instr, sa);
        Ok(())
    }

    fn step(&mut self, u: &mut Unstructured) -> Result<(), ArbError> {
        // 25% leaf (if possible), else op
        let roll: u8 = u.int_in_range(0..=99)?;
        if roll < 25 && self.leaf_count < self.cfg.max_leaves {
            self.gen_leaf(u)
        } else {
            // unary/binary only for now (safe across ranks)
            let op_roll: u8 = u.int_in_range(0..=99)?;
            if op_roll < 50 { self.gen_unary(u) } else { self.gen_binary(u) }
        }
    }

    fn build_autograd(self) -> AutogradProgram {
        AutogradProgram {
            rows: 1,
            cols: 1,
            leaf_seeds: self.leaf_seeds,
            ops: self.ops,
            shapes: self.arena,
        }
    }

    fn build_tensor(self) -> TensorProgram {
        // TensorProgram uses only TensorInstr ops (no Leaf ops)
        let ops: Vec<TensorInstr> = self.ops.into_iter().filter_map(|op| {
            match op {
                DiffOp::Instr(i) => Some(i),
                _ => None,
            }
        }).collect();

        TensorProgram {
            rows: 1,
            cols: 1,
            values: self.leaf_seeds.first().cloned().unwrap_or_default(),
            ops,
            shapes: self.arena,
        }
    }
}

impl<'a> Arbitrary<'a> for AutogradProgram {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self, ArbError> {
        let cfg = read_cfg();
        let mut b = Builder::new(cfg.clone());

        // r0 seed leaf
        let sh = sample_shape_1to4(u, &cfg)?;
        let seed = seed_bytes(u)?;
        b.add_seed_leaf(sh, seed);

        let steps: usize = u.int_in_range(1..=48)?;
        for _ in 0..steps {
            if u.is_empty() { break; }
            b.step(u)?;
        }
        Ok(b.build_autograd())
    }
}

impl<'a> Arbitrary<'a> for TensorProgram {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self, ArbError> {
        let cfg = read_cfg();
        let mut b = Builder::new(cfg.clone());

        // r0
        let sh = sample_shape_1to4(u, &cfg)?;
        let seed = seed_bytes(u)?;
        b.add_seed_leaf(sh, seed);

        // tensor program = ops only
        let steps: usize = u.int_in_range(1..=48)?;
        for _ in 0..steps {
            if u.is_empty() { break; }
            // only generate unary/binary ops
            let op_roll: u8 = u.int_in_range(0..=99)?;
            if op_roll < 50 { b.gen_unary(u)?; } else { b.gen_binary(u)?; }
        }
        Ok(b.build_tensor())
    }
}