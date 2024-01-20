use crate::{op::Op, tensor_data::TensorData, Tensor};

pub fn relu(x: Tensor) -> Tensor {
    let mut data = x.item();
    for item in data.iter_mut() {
        *item = if *item > 0.0 { *item } else { 0.0 }
    }
    let shape = x.shape();
    let inner = TensorData::from_op(data, vec![x], Op::ReLU);
    Tensor::new(inner, &shape)
}

pub fn sigmoid(x: Tensor) -> Tensor {
    let data = ((-x.clone()).exp() + 1.0).pow(-1);
    let inner = TensorData::from_op(data.item(), vec![x.clone()], Op::Sigmoid(x));
    Tensor::new(inner, &data.shape)
}
