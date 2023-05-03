#[derive(Clone)]
pub enum Op {
    Add,
    Mul,
    Pow(i32),
    ReLU,
    Sigmoid(f64),
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "Add"),
            Op::Mul => write!(f, "Mul"),
            Op::Pow(n) => write!(f, "Pow({n})"),
            Op::ReLU => write!(f, "ReLU"),
            Op::Sigmoid(n) => write!(f, "Sigmoid({n})"),
        }
    }
}
