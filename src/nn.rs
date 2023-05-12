use crate::{op::Op, tensor_data::TensorData, Tensor};

pub trait Module {
    fn module_name(&self) -> String;

    fn forward(&self, x: Tensor) -> Tensor;
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
}

pub fn relu(x: Tensor) -> Tensor {
    let mut data = x.item();
    for i in 0..data.len() {
        data[i] = if data[i] > 0.0 { data[i] } else { 0.0 }
    }
    let backward = |_tensor| todo!();
    let inner = TensorData::from_op(data, x.shape(), vec![x], backward, Op::ReLU);
    Tensor::new(inner)
}

pub fn sigmoid(x: Tensor) -> Tensor {
    (x.exp() + 1.0).pow(-1)
}
