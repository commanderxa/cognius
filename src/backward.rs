use std::cell::Ref;

use crate::{op::Op, tensor_data::TensorData};

/// Backward trait for backpropagation operation.
pub trait Backward {
    fn backward(&self, tensor: &Ref<TensorData>);
}

/// `Backward` trait implemeted for `Op` where eacj operation has it's own
/// backpropagation operation.
impl Backward for Op {
    fn backward(&self, tensor: &Ref<TensorData>) {
        match self {
            // Addition backward
            // 1.0 * grad for both previous tensors
            Op::Add => {
                let t = tensor;
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_to_grad(grad.clone());
                t._prev[1].add_to_grad(grad);
            }

            // Multiplication backward
            Op::Mul => {
                let t = tensor;
                let grad = t.grad.clone().unwrap();
                let l_item = t._prev[0].item();
                let r_item = t._prev[1].item();
                t._prev[0].add_to_grad(
                    r_item
                        .iter()
                        .zip(grad.clone())
                        .map(|(a, b)| a * b)
                        .collect(),
                );
                t._prev[1].add_to_grad(l_item.iter().zip(grad).map(|(a, b)| a * b).collect());
            }

            // Power backward
            // d(x^n)/dx * grad = n * x^(n-1) * grad
            Op::Pow(n) => {
                let n = n.clone();
                let t = tensor;
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_to_grad(
                    t.data
                        .iter()
                        .zip(grad)
                        .map(|(x, y)| n as f64 * x.powi(n - 1) * y)
                        .collect(),
                );
            }
            // Exponent backward
            Op::Exp(t) => {
                let grad = tensor.grad.clone().unwrap();
                tensor._prev[0].add_to_grad(
                    t.exp()
                        .item()
                        .iter()
                        .zip(grad)
                        .map(|(a, b)| a * b)
                        .collect(),
                );
            }

            // Matrix Multiplication backward
            Op::MatMul => todo!(),

            // ReLU backward
            Op::ReLU => {
                let t = tensor;
                let mut prev = t._prev[0].0.borrow_mut();
                let grad = prev.grad.as_mut().unwrap();
                for i in 0..t.data.len() {
                    grad[i] = if t.data[i] > 0.0 { 1.0 } else { 0.0 }
                }
            }

            // Sigmoid backward
            Op::Sigmoid(_) => {
                // sigmoid function:        1 / (1 + exp(-x))
                // dx(sigmoid) function:    exp(-x) / (1 + exp(-x))^2
            }
        }
    }
}
