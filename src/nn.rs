pub mod optim;

use crate::{op::Op, tensor_data::TensorData, Tensor};

pub trait Module {
    fn module_name(&self) -> String;

    fn forward(&self, x: Tensor) -> Tensor;

    fn parameters(&self) -> Vec<Tensor>;
}

pub struct Linear {
    pub weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let _weight = Tensor::randn(vec![in_features, out_features]);
        let _bias = Tensor::randn(vec![out_features, 1]);
        Self {
            weight: _weight,
            bias: None,
        }
    }

    pub fn no_bias(mut self) -> Self {
        self.bias = None;
        self
    }
}

impl Module for Linear {
    fn module_name(&self) -> String {
        "Linear".to_owned()
    }

    fn forward(&self, x: Tensor) -> Tensor {
        let weight = self.weight.clone();
        let bias = self.bias.clone();
        let mut x = Tensor::mm(x, weight);
        if let Some(b) = bias {
            x = x + b;
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = vec![self.weight.clone()];
        if let Some(b) = self.bias.clone() {
            parameters.push(b);
        }
        parameters
    }
}

pub fn relu(x: Tensor) -> Tensor {
    let mut data = x.item();
    for item in data.iter_mut() {
        *item = if *item > 0.0 { *item } else { 0.0 }
    }
    let inner = TensorData::from_op(data, x.shape(), vec![x], Op::ReLU);
    Tensor::new(inner)
}

pub fn sigmoid(x: Tensor) -> Tensor {
    let data = ((-x.clone()).exp() + 1.0).pow(-1);
    let inner = TensorData::from_op(data.item(), data.shape(), vec![x.clone()], Op::Sigmoid(x));
    Tensor::new(inner)
}

pub struct MSELoss {}

impl MSELoss {
    pub fn new() -> Self {
        Self {}
    }

    pub fn measure(&self, a: Tensor, b: Tensor) -> Tensor {
        let t = (a - b).pow(2);
        let inner = TensorData::from_op(t.item(), t.shape(), vec![t], Op::MSE);
        Tensor::new(inner)
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}
