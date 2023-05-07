/// # TensorData
/// 
/// ### This type holds the data of the tensor, and is an actual data.
/// 
/// `Tensor` holds a reference to it, allowing to use the same data.\
/// See the documentation for `Tensor`.
pub struct TensorData {
    pub data: Vec<f64>,
    pub grad: Vec<f64>,
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Create a new instance of the TensorData.
    pub fn new(dims: Vec<usize>) -> Self {
        let len = dims.iter().product();
        Self {
            data: vec![0.0; len],
            grad: vec![0.0; len],
            shape: dims,
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
            grad: grad,
            shape: shape,
        }
    }
}
