use crate::{
    linalg,
    module::{Forward, Module},
    Tensor,
};

/// # `Linear` Layer
///
/// Contains of:
/// - weights
/// - bias
///
/// Linear layer performs: `x @ W + b`,
/// where:
/// - `x` is input
/// - `W` is weights
/// - `b` is bias
pub struct Linear {
    pub weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let _weight = Tensor::randn(&[in_features, out_features]);
        let _bias = Tensor::randn(&[out_features, 1]);
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

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = vec![self.weight.clone()];
        if let Some(b) = self.bias.clone() {
            parameters.push(b);
        }
        parameters
    }
}

impl Forward for Linear {
    fn forward(&self, x: Tensor) -> Tensor {
        let weight = self.weight.clone();
        let bias = self.bias.clone();
        let mut x = linalg::matmul(x, weight);
        if let Some(b) = bias {
            x = x + b;
        }
        x
    }
}
