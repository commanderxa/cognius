#[derive(Clone, Debug, PartialEq)]
/// Operations that are available to apply to `Value`.
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

// Alternative Backward idea
//
//
//
// pub trait Backward {
//     fn backward(&self, tensor: Tensor);
// }

// impl Backward for Op {
//     fn backward(&self, tensor: Tensor) {
//         match self {
//             Op::Add => {
//                 let t = tensor.0.borrow();
//                 let grad = t.grad.clone().unwrap();
//                 t._prev[0].add_grad(grad.clone());
//                 t._prev[1].add_grad(grad);
//             },
//             Op::Mul => todo!(),
//             Op::Pow(_) => todo!(),
//             Op::ReLU => todo!(),
//             Op::Sigmoid(_) => todo!(),
//         }
//     }
// }
