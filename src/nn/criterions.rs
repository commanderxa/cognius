use crate::{op::Op, tensor_data::TensorData, Tensor};

pub struct MSELoss {}

impl MSELoss {
    pub fn new() -> Self {
        Self {}
    }

    pub fn measure(&self, a: Tensor, b: Tensor) -> Tensor {
        let t = (a - b).pow(2);
        let shape = t.shape();
        let inner = TensorData::from_op(t.item(), vec![t], Op::MSE);
        Tensor::new(inner, &shape)
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}
