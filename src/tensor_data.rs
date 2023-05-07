use crate::{op::Op, Tensor};

/// # TensorData
///
/// ### This type holds the data of the tensor, and is an actual data.
///
/// `Tensor` holds a reference to it, allowing to use the same data.\
/// See the documentation for `Tensor`.
pub struct TensorData {
    // stored data
    // it is a single vector that is viewed regarding the shape
    pub data: Vec<f64>,
    // gradients
    pub grad: Option<Vec<f64>>,
    // shape of the tensor
    pub shape: Vec<usize>,
    // vector of the tensors that were used to produce this tensor
    pub _prev: Vec<Tensor>,
    // backward function
    pub _backward: Option<fn(tensor: Tensor)>,
    // operation that was used to produce this tensor
    pub _op: Option<Op>,
}

impl TensorData {
    /// Create a new instance of the TensorData.
    pub fn new(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            grad: Some(vec![0.0; len]),
            shape: shape,
            _prev: vec![],
            _backward: None,
            _op: None,
        }
    }

    /// Sets the gradients to 0.
    pub fn fill_grad(tensor: &mut Vec<f64>) {
        for _ in 0..tensor.len() {
            tensor.push(0.0);
        }
    }

    /// Creates a new instance of the TensorData from a Vector.
    pub fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let mut grad = Vec::with_capacity(data.len());
        Self::fill_grad(&mut grad);
        Self {
            data: data,
            grad: Some(grad),
            shape: shape,
            _prev: vec![],
            _backward: None,
            _op: None,
        }
    }

    pub fn from_op(
        data: Vec<f64>,
        shape: Vec<usize>,
        prev: Vec<Tensor>,
        backward: fn(Tensor),
        op: Op,
    ) -> Self {
        let mut grad = Vec::with_capacity(data.len());
        Self::fill_grad(&mut grad);
        Self {
            data: data,
            grad: Some(grad),
            shape: shape,
            _prev: prev,
            _backward: Some(backward),
            _op: Some(op),
        }
    }
}
