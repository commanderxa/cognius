use crate::Tensor;

#[derive(Clone, Debug, PartialEq)]
/// Operations that are available to apply to `Value`.
pub enum Op {
    Add,
    Sub,
    Mul,
    Pow(i32),
    Exp(Tensor),
    MatMul,
    Cross,
    ReLU,
    Sigmoid(Tensor),
    #[allow(clippy::upper_case_acronyms)]
    MSE,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "Add"),
            Op::Sub => write!(f, "Sub"),
            Op::Mul => write!(f, "Mul"),
            Op::Pow(n) => write!(f, "Pow({n})"),
            Op::Exp(_) => write!(f, "Exp"),
            Op::MatMul => write!(f, "MatMul"),
            Op::Cross => write!(f, "Cross"),
            Op::ReLU => write!(f, "ReLU"),
            Op::Sigmoid(n) => write!(f, "Sigmoid({n})"),
            Op::MSE => write!(f, "MSE"),
        }
    }
}
