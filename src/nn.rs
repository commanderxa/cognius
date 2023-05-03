use rand::Rng;

use crate::Value;

/// Implementation of Neuron
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    /// Creates an instance of a `Neuron`.
    /// 
    /// Accepts:
    /// * in_features: positive number
    /// 
    /// `in_features` is the number of `Value`s that the `Neuron` will have.
    pub fn new(in_features: usize) -> Self {
        let mut weights = Vec::with_capacity(in_features);
        for _ in 0..in_features {
            let data = rand::thread_rng().gen_range(-1.0..1.0);
            let value = Value::from(data);
            weights.push(value);
        }
        Self {
            w: weights,
            b: Value::from(0.0),
            nonlin: false,
        }
    }

    /// Sets the nonlinearity for a `Neuron`.
    /// 
    /// Accepts:
    /// * value: boolean
    pub fn set_nonlin(&mut self, value: bool) {
        self.nonlin = value;
    }

    /// Inference method for a `Neuron`.
    /// 
    /// Accpets:
    /// * x: Vector of `Value`
    /// 
    /// Returns `Value`.
    pub fn call(&self, x: Vec<Value>) -> Value {
        let mut result = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            let y = wi.clone() * xi.clone();
            result = result + y;
        }
        if self.nonlin {
            result.sigmoid()
        } else {
            result
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut parameters = self.w.clone();
        parameters.push(self.b.clone());
        parameters
    }
}

/// Linear layer implementation
/// 
/// `Linear` consists of `Neuron`s.\
/// See the documentation for `Neuron`.
pub struct Linear {
    neurons: Vec<Neuron>,
}

impl Linear {

    /// Creates an instance of `Linear`.
    /// 
    /// Accepts:
    /// * in_features: positive number
    /// * out features: positive number
    /// 
    /// `in_features` is the number of features that will be passed to the `Linear`.\
    /// `out_features` is the number of features to be oututed from the `Linear` layer.
    pub fn new(in_features: usize, out_features: usize, nonlin: bool) -> Self {
        let mut neurons = Vec::with_capacity(in_features);
        for _ in 0..out_features {
            let mut neuron = Neuron::new(in_features);
            if nonlin {
                neuron.set_nonlin(true);
            }
            neurons.push(neuron);
        }
        Self { neurons }
    }

    pub fn call(&self, x: Vec<Value>) -> Vec<Value> {
        let mut result = Vec::new();
        for n in &self.neurons {
            result.push(n.call(x.clone()));
        }
        result
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Value> {
        let mut parameters = Vec::new();
        for n in &self.neurons {
            for p in n.parameters() {
                parameters.push(p);
            }
        }
        parameters
    }
}

/// Multi Layer Perceptron (MLP)
/// 
/// `MLP` consists of `Linear` layers.\
/// See the documentation for `Linear`.
pub struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    /// Creates a new instance of `MLP`.
    /// 
    /// Accepts:
    /// * in_features: positive number
    /// * out features: Vector of positive numbers
    /// 
    /// `in_features` is the number of features that will be passed to the `MLP`.\
    /// `out_features` is the vector of features to be oututed from each layer.
    pub fn new(in_features: usize, out_features: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let sz: Vec<usize> = vec![in_features]
            .into_iter()
            .chain(out_features.into_iter())
            .collect();
        for i in 0..sz.len() {
            let nonlin = i != sz.len() - 1;
            layers.push(Linear::new(i, i + 1, nonlin));
        }
        Self { layers }
    }
    
    /// Inference method for MLP.
    /// 
    /// Accepts:
    /// * x: Vector of `Value`
    /// 
    /// Returns a vector of `Value`.
    pub fn call(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.call(x)
        }
        x
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        let mut parameters = Vec::new();
        for layer in &self.layers {
            for p in layer.parameters() {
                parameters.push(p);
            }
        }
        parameters
    }
}

pub trait Module {

    /// Returns the parameters of the model.
    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.set_grad(0.0);
        }
    }
}
