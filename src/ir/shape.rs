use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeN {
    pub rank: u8,            // 1..=4
    pub dims: [usize; 4],    // only first `rank` entries are used
}

impl ShapeN {
    pub fn new(rank: u8, dims: [usize; 4]) -> Self {
        let r = rank.clamp(1, 4);
        let mut d = dims;
        for i in 0..4 {
            d[i] = d[i].max(1);
        }
        // zero out unused dims for cleanliness
        for i in (r as usize)..4 {
            d[i] = 1;
        }
        Self { rank: r, dims: d }
    }

    pub fn numel(&self) -> usize {
        let r = self.rank as usize;
        let mut n = 1usize;
        for i in 0..r {
            n = n.saturating_mul(self.dims[i]);
        }
        n
    }

    /// If rank >= 2, treat dims[0],dims[1] as rows,cols for display convenience.
    pub fn rows(&self) -> usize { self.dims[0] }
    pub fn cols(&self) -> usize { if self.rank >= 2 { self.dims[1] } else { 1 } }
}

impl fmt::Display for ShapeN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.rank as usize;
        write!(f, "[")?;
        for i in 0..r {
            if i > 0 { write!(f, "×")?; }
            write!(f, "{}", self.dims[i])?;
        }
        write!(f, "]")
    }
}