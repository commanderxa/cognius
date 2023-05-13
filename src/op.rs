use crate::Tensor;

#[derive(Clone, Debug, PartialEq)]
/// Operations that are available to apply to `Value`.
pub enum Op {
    Add,
    Mul,
    Pow(i32),
    Exp(Tensor),
    MatMul,
    ReLU,
    Sigmoid(f64),
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "Add"),
            Op::Mul => write!(f, "Mul"),
            Op::Pow(n) => write!(f, "Pow({n})"),
            Op::Exp(_) => write!(f, "Exp(..)"),
            Op::MatMul => write!(f, "MatMul"),
            Op::ReLU => write!(f, "ReLU"),
            Op::Sigmoid(n) => write!(f, "Sigmoid({n})"),
        }
    }
}
