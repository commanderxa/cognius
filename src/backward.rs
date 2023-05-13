use crate::{Tensor, op::Op};

/// Backward trait for backpropagation operation.
pub trait Backward {
    fn backward(&self, tensor: Tensor);
}

/// `Backward` trait implemeted for `Op` where eacj operation has it's own 
/// backpropagation operation.
impl Backward for Op {
    fn backward(&self, tensor: Tensor) {
        match self {
            Op::Add => {
                let t = tensor.0.borrow();
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_grad(grad.clone());
                t._prev[1].add_grad(grad);
            },
            Op::Mul => todo!(),
            Op::Pow(_) => todo!(),
            Op::Exp(_) => todo!(),
            Op::MatMul => todo!(),
            Op::ReLU => todo!(),
            Op::Sigmoid(_) => todo!(),
        }
    }
}