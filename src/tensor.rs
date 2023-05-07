use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul, Neg, Range},
    rc::Rc,
};

use rand::Rng;

pub struct Tensor {
    data: Rc<RefCell<TensorData>>,
}

impl Tensor {
    /// Creates a new tensor with the random values between 0 and 1
    pub fn randn(dims: Vec<usize>) -> Self {
        let mut tensor_data = TensorData::from_f64(vec![], dims);
        Self::fill_tensor(&mut tensor_data, 0.0..1.0);
        let tensor = Rc::new(RefCell::new(tensor_data));
        Self { data: tensor }
    }

    /// Creates a new tensor, where all the values are 0.
    pub fn zeros(dims: Vec<usize>) -> Self {
        let mut tensor_data = TensorData::new(dims);
        tensor_data.data.fill(0.0);
        let tensor = Rc::new(RefCell::new(tensor_data));
        Self { data: tensor }
    }

    /// Creates a new tensor, where all the values are 1.
    pub fn ones(dims: Vec<usize>) -> Self {
        let mut tensor_data = TensorData::new(dims);
        tensor_data.data.fill(1.0);
        let tensor = Rc::new(RefCell::new(tensor_data));
        Self { data: tensor }
    }

    /// Create a new tensor from the given data and the shape.
    pub fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "The length of the tensor does not match the shape"
        );
        let tensor_data = TensorData::from_f64(data, shape);
        let tensor = Rc::new(RefCell::new(tensor_data));
        Self { data: tensor }
    }

    /// Generates a tesnor within a given range with a step.
    ///
    /// The `start` is inclusive and the `end` is exclusive.
    /// 
    /// Note: 
    /// * (`end` - `start`) / `step` has to be an integer
    /// * if `end` - `start` is negative, then `step` has to be negative
    /// * if `end` - `start` is positive, then `step` has to be positive
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let len = (end - start) / step;
        // necessary cheks
        assert_eq!(len.fract(), 0.0, "Cannot generate a range, since the length of the tensor is not an integer, try to use other parameters");
        if end - start < 1.0 && step > 0.0 {
            panic!("Cannot generate a range, since the step is wrong, try to make it negative");
        } else if end - start > 1.0 && step < 0.0 {
            panic!("Cannot generate a range, since the step is wrong, try to make it positive");
        }
        // new tensor
        let mut data = Vec::with_capacity(len as usize);
        for i in 0..len as usize {
            data.push(start + (step * i as f64));            
        }
        let tensor_data = TensorData::from_f64(data, vec![len as usize]);
        let tensor = Rc::new(RefCell::new(tensor_data));
        Self { data: tensor }
    }

    /// Returns the shape of the tensor as vector.
    pub fn shape(&self) -> Vec<usize> {
        self.data.borrow().shape.clone()
    }

    /// Returns the total length of the vector. It obtains the length by taking the
    /// product of the shape of the tensor.
    ///
    /// E.g. if the shape of the tensor is (3, 2, 3), then the length is (3 * 2 * 3)
    /// => 18.
    pub fn length(&self) -> usize {
        self.shape().iter().product()
    }

    /// Fills the given empty tensor with a values with an inputted range. It also
    /// sets the gradients to 0.
    ///
    /// E.g. if the range is (-1.0..1.0) then each value in the tensor will be
    /// between -1 and 1.
    fn fill_tensor(tensor: &mut TensorData, range: Range<f64>) {
        for _ in 0..tensor.shape.iter().product() {
            let data = rand::thread_rng().gen_range(range.clone());
            tensor.data.push(data);
            tensor.grad.push(0.0);
        }
    }

    /// Returns the data of the tensor.
    pub fn item(&self) -> Vec<f64> {
        self.data.borrow().data.clone()
    }

    /// Transpose
    ///
    /// This method transposes the tensor, it changes the shape.
    /// The rows become the columns and vice versa.
    ///
    /// E.g. if the shape was (2, 3) it will make (3, 2).
    pub fn t(&self) -> Self {
        // borrow data for easy access
        // let shape = &self.data.borrow().shape;
        let tensor = &self.data.borrow().data;
        let mut shape = self.shape();
        // new data vector
        let mut data = Vec::with_capacity(self.length());
        // iterate over the tensor by column
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                data.push(tensor[i + (j * shape[0])]);
            }
        }
        // reverse the shape
        shape.reverse();
        // create a new tensor
        let inner = TensorData::from_f64(data, shape);
        Self {
            data: Rc::new(RefCell::new(inner)),
        }
    }

    pub fn mm(self: Self, rhs: Self) -> Self {
        // shapes of the tensors
        let left_shape = self.shape();
        let right_shape = rhs.shape();
        // new tensor result
        let result = Tensor::ones(vec![left_shape[0], right_shape[right_shape.len() - 1]]);
        // check wether the operation is able to be proceeded
        assert_eq!(
            left_shape.last().unwrap(), 
            right_shape.first().unwrap(), 
            "The shapes of the tensors must have the same inner dimension -> (M x N) @ (N x M), but you have tensors A: {:?} and B: {:?}", 
            left_shape, 
            right_shape
        );
        // data of the tensors, the tensor on the right is transposed
        let left = self.item();
        let right = rhs.t().item();

        // iterate over the result tensor, it zips the slices of the left and right tensors
        // then it multiplies the two zipped values and returns the slice back, after it sums
        // the vector to obtain the value
        for i in 0..left_shape[0] {
            for j in 0..right_shape[1] {
                let right = &right[(j * left_shape[1])..(j * left_shape[1] + left_shape[1])];
                result.data.borrow_mut().data[i * left_shape[0] + j] = left
                    [(i * left_shape[1])..(i * left_shape[1] + left_shape[1])]
                    .iter()
                    .zip(right)
                    .map(|(&a, &b)| a * b)
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();
            }
        }
        result
    }

    fn tensor_to_str(&self, tensor_str: String, level: usize, range: Range<usize>) -> String {
        // the length of the range
        let len: usize = range.end - range.start;
        // the current dimension from the shape
        let dim = self.shape()[level];
        // convolution to iterate over the data
        let conv = len / dim;
        // the length of shape vector
        let shape_size = self.shape().len();

        // denote the start of the dimension
        let mut result = String::from("[");
        // iterate over the dimension
        for i in (range.start..range.end).step_by(conv) {
            if shape_size - 1 == level {
                let mut num = format!("{:.4}", self.data.borrow().data[i]);
                if i < self.shape()[level] - 1 {
                    num.push_str(", ");
                }
                result.push_str(num.as_str());
            }
            // if the iteration is over the last 2 dimensions => produce a matrix
            else if shape_size - 2 == level {
                result.push_str("[");
                for j in 0..self.shape()[level + 1] {
                    let mut num = format!("{:.4}", self.data.borrow().data[i + j]);
                    if j < self.shape()[level + 1] - 1 {
                        num.push_str(", ");
                    }
                    result.push_str(num.as_str());
                }
                // close the matrix and add indents for the following row (if exists)
                if i != range.end - conv {
                    result.push_str("],\n\t");
                    let space = String::from(" ").repeat(shape_size - 2);
                    result.push_str(space.as_str());
                } else {
                    result.push_str("]");
                }
            } else {
                // else, fall further into the next dimensions
                result.push_str(
                    self.tensor_to_str(tensor_str.clone(), level + 1, i..(i + conv))
                        .as_str(),
                );
                // make indents for following tensor (if exists)
                if i != range.end - conv {
                    result.push_str(",\n\n\t");
                    let space = String::from(" ").repeat(level);
                    result.push_str(space.as_str());
                }
            }
        }
        // denote the end of the dimension
        result.push_str("]");
        result
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        if rhs.shape().iter().product::<usize>() == 1 {
            for i in 0..self.shape().iter().product() {
                self.data.borrow_mut().data[i] += rhs.data.borrow_mut().data[0];
            }
        } else {
            assert_eq!(self.shape(), rhs.shape());
            let shape = self.shape();
            for i in 0..shape.iter().product() {
                self.data.borrow_mut().data[i] += rhs.data.borrow().data[i];
            }
        }
        self
    }
}

impl Add<i64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: i64) -> Self::Output {
        for i in 0..self.shape().iter().product() {
            self.data.borrow_mut().data[i] += rhs as f64;
        }
        self
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        for i in 0..self.shape().iter().product() {
            self.data.borrow_mut().data[i] += rhs;
        }
        self
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.length() == 1 {
            // just multiply each value by the single value in the right tensor
            for i in 0..self.shape().iter().product() {
                self.data.borrow_mut().data[i] *= rhs.data.borrow_mut().data[0];
            }
        } else {
            todo!()
        }
        self
    }
}

impl Mul<i64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: i64) -> Self::Output {
        for i in 0..self.shape().iter().product() {
            self.data.borrow_mut().data[i] *= rhs as f64;
        }
        self
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        for i in 0..self.shape().iter().product() {
            self.data.borrow_mut().data[i] *= rhs;
        }
        self
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let negative = Tensor::from_f64(vec![-1.0], vec![1]);
        self * negative
    }
}

struct TensorData {
    data: Vec<f64>,
    grad: Vec<f64>,
    shape: Vec<usize>,
}

impl TensorData {
    fn new(dims: Vec<usize>) -> Self {
        let len = dims.iter().product();
        Self {
            data: vec![0.0; len],
            grad: vec![0.0; len],
            shape: dims,
        }
    }

    /// Sets the gradients to 0.
    fn fill_grad(tensor: &mut Vec<f64>) {
        for _ in 0..tensor.len() {
            tensor.push(0.0);
        }
    }

    fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let mut grad = Vec::with_capacity(data.len());
        Self::fill_grad(&mut grad);
        Self {
            data: data,
            grad: grad,
            shape: shape,
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = String::new();
        let dims = self.shape();
        let data = self.tensor_to_str(data, 0, 0..dims.iter().product());

        let mut dims_str = String::from("(");
        for (i, dim) in dims.iter().enumerate() {
            dims_str.push_str(format!("{dim}").as_str());
            if i < dims.len() - 1 {
                dims_str.push_str(", ");
            }
        }
        dims_str.push_str(")");

        let res = format!("Tensor({data}, shape: {dims_str}, dtype: f64)");
        write!(f, "{res}")
    }
}
