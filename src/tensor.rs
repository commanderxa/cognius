use std::{
    cell::RefCell,
    f64::consts::E,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Range, Sub},
    rc::Rc,
};

use rand::Rng;

use crate::{backward::Backward, op::Op, tensor_data::TensorData};

#[derive(Clone, Debug, PartialEq)]
/// # Tensor
///
/// ### Holds the reference to the inner data inside.
///
/// See the documentation for `TensorData`.
pub struct Tensor(pub(crate) Rc<RefCell<TensorData>>);

impl Tensor {
    /// Creates a new instance of a `Tensor`.
    pub(crate) fn new(inner: TensorData) -> Self {
        Self(Rc::new(RefCell::new(inner)))
    }

    /// Creates a new tensor with the random values between 0 and 1
    pub fn randn(shape: Vec<usize>) -> Self {
        let mut inner = TensorData::from_f64(vec![], shape);
        Self::fill_tensor(&mut inner, 0.0..1.0);
        Self::new(inner)
    }

    /// Creates a new tensor, where all the values are 0.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let mut inner = TensorData::new(shape);
        inner.data.fill(0.0);
        Self::new(inner)
    }

    /// Creates a new tensor like the inputted one, where all the values are 0.
    pub fn zeros_like(tensor: Tensor) -> Self {
        let mut inner = TensorData::new(tensor.shape());
        inner.data.fill(0.0);
        Self::new(inner)
    }

    /// Creates a new tensor, where all the values are 1.
    pub fn ones(shape: Vec<usize>) -> Self {
        let mut inner = TensorData::new(shape);
        inner.data.fill(1.0);
        Self::new(inner)
    }

    /// Creates a new tensor like the inputted one, where all the values are 1.
    pub fn ones_like(tensor: Tensor) -> Self {
        let mut inner = TensorData::new(tensor.shape());
        inner.data.fill(1.0);
        Self::new(inner)
    }

    /// Create a new tensor from the given data and the shape.
    pub fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "The length of the tensor does not match the shape"
        );
        let inner = TensorData::from_f64(data, shape);
        Self::new(inner)
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
        assert_eq!(
            len.fract(),
            0.0,
            "Cannot generate a range, since the length of the tensor is not an integer, try to use other parameters"
        );
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
        let inner = TensorData::from_f64(data, vec![len as usize]);
        Self::new(inner)
    }

    /// Returns the shape of the tensor as vector.
    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }

    /// Returns the total length of the vector. It obtains the length by taking
    /// the product of the shape of the tensor.
    ///
    /// E.g. if the shape of the tensor is (3, 2, 3), then the length is
    /// (3 * 2 * 3) => 18.
    pub fn length(&self) -> usize {
        self.shape().iter().product()
    }

    /// Fills the given empty tensor with a values with an inputted range. It
    /// also sets the gradients to 0.
    ///
    /// E.g. if the range is (-1.0..1.0) then each value in the tensor will be
    /// between -1 and 1.
    fn fill_tensor(tensor: &mut TensorData, range: Range<f64>) {
        for _ in 0..tensor.shape.iter().product() {
            let data = rand::thread_rng().gen_range(range.clone());
            tensor.data.push(data);
            tensor.grad.as_mut().unwrap().push(0.0);
        }
    }

    /// Returns the data of the tensor.
    pub fn item(&self) -> Vec<f64> {
        self.0.borrow().data.clone()
    }

    /// Defines the `Tensor` behavior.
    ///
    /// The tensor property `requires_grad` is `true` by default, which means
    /// that the `Tensor` has a gradient, but this gradient might be sent to
    /// `None` if is not necessary.
    pub fn requires_grad(self, value: bool) -> Self {
        if value {
            self.0.borrow_mut().grad = Some(vec![0.0; self.length()]);
        } else {
            self.0.borrow_mut().grad = None;
        }
        self
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
        let tensor = &self.0.borrow().data;
        let mut shape = self.shape();
        // new data vector
        let mut data = Vec::with_capacity(self.length());
        // iterate over the tensor by column
        for i in 0..shape[1] {
            for j in 0..shape[0] {
                data.push(tensor[(j * shape[1]) + i]);
            }
        }
        // reverse the shape
        shape.reverse();
        // create a new tensor
        let inner = TensorData::from_f64(data, shape);
        Self::new(inner)
    }

    /// Matrix multiplication
    ///
    /// Accepts:
    /// * a: `Tensor`
    /// * b: `Tensor`
    ///
    /// The inner dimensions of the matrices must be the same.
    pub fn mm(a: Self, b: Self) -> Self {
        // shapes of the tensors
        let a_shape = a.shape();
        let b_shape = b.shape();
        // new tensor result
        let mut result = vec![0.0; a_shape[0] * (b_shape[b_shape.len() - 1])];
        // check wether the operation is able to be proceeded
        assert_eq!(
            a_shape.last().unwrap(),
            b_shape.first().unwrap(),
            "The shapes of the tensors must have the same inner dimension -> (M x N) @ (N x M), but you have tensors A: {:?} and B: {:?}", 
            format!("({a_shape:?})").replace("[", "").replace("]", ""), 
            format!("({b_shape:?})").replace("[", "").replace("]", ""), 
        );
        // data of the tensors, the tensor b is transposed
        let a_data = a.item();
        let b_data = b.t().item();

        // iterate over the result tensor, it zips the slices of the left and
        // right tensors then it multiplies the two zipped values and returns
        // the slice back, after it sums the vector to obtain the value
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                let b_data = &b_data[(j * a_shape[1])..(j * a_shape[1] + a_shape[1])];
                result[b_shape[1] * i + j] = a_data
                    [(i * a_shape[1])..(i * a_shape[1] + a_shape[1])]
                    .iter()
                    .zip(b_data)
                    .map(|(&a, &b)| a * b)
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();
            }
        }
        let inner = TensorData::from_op(
            result,
            vec![a_shape[0], b_shape[b_shape.len() - 1]],
            vec![a, b],
            Op::MatMul,
        );
        Tensor::new(inner)
    }

    /// Returns the tensor of the new shape.
    ///
    /// Tensor of shape (2, 3) might be viewed as (3, 2), (6, 1), (1, 6).
    /// Tensor can be viewed as any shape, only if the length of this shape is
    /// the same as the length of the previous shape.
    pub fn view(&self, shape: Vec<usize>) -> Self {
        assert_eq!(
            self.length(),
            shape.iter().product(),
            "Length of the new shape: {} does not match the length of the old one: {}",
            shape.iter().product::<usize>(),
            self.length()
        );
        let inner = TensorData::from_f64(self.item(), shape);
        Self::new(inner)
    }

    /// Reshapes the tensor.
    ///
    /// Tensor of shape (2, 3) might be reshaped to (3, 2), (6, 1), (1, 6).
    /// Tensor can be reshaped into any shape, only if the length of this shape
    /// is the same as the length of the previous shape.
    pub fn reshape(&self, shape: Vec<usize>) {
        assert_eq!(
            self.length(),
            shape.iter().product(),
            "Length of the new shape: {} does not match the length of the old one: {}",
            shape.iter().product::<usize>(),
            self.length()
        );
        self.0.borrow_mut().shape = shape;
    }

    /// Exponents each value of the `Tensor`.
    ///
    /// `exp(x)` => `e^(x)`.
    pub fn exp(&self) -> Tensor {
        let mut data = self.item();
        for i in 0..data.len() {
            data[i] = E.powf(data[i]);
        }
        let inner = TensorData::from_op(
            data,
            self.shape(),
            vec![self.clone()],
            Op::Exp(self.clone()),
        );
        Tensor::new(inner)
    }

    /// Converts the tensor to a `String`, so that it can be printed.
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
        // iterate over the dimension => print a vector
        for i in (range.start..range.end).step_by(conv) {
            // if the dimension is the last one
            if shape_size - 1 == level {
                let mut num = format!("{:.4}", self.0.borrow().data[i]);
                if i < self.shape()[level] - 1 {
                    num.push_str(", ");
                }
                result.push_str(num.as_str());
            }
            // if the iteration is over the last 2 dimensions => print a matrix
            else if shape_size - 2 == level {
                result.push_str("[");
                for j in 0..self.shape()[level + 1] {
                    let mut num = format!("{:.4}", self.0.borrow().data[i + j]);
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

    /// Add a `Vec<f64>` value to the gradient inside the `TensorData`.
    pub(crate) fn add_to_grad(&self, data: Vec<f64>) {
        let mut t = self.0.borrow_mut();
        t.grad = Some(
            t.grad
                .clone()
                .unwrap()
                .iter()
                .zip(data)
                .map(|(a, b)| a + b)
                .collect(),
        );
    }

    /// Returns the gradient vector.
    pub fn grad(&self) -> Option<Vec<f64>> {
        self.0.borrow().grad.clone()
    }

    pub fn set_data(&self, data: Vec<f64>) {
        self.0.borrow_mut().data = data;
    }

    /// Powers the `Tensor`
    ///
    /// Accepts `n` integer in which the `Tensor` will be powered.
    ///
    /// For backpropagation it stores the `n` inside the `Op::Pow(n)`.
    pub fn pow(self, n: i32) -> Tensor {
        let data = self.item().iter().map(|a| a.powi(n)).collect::<Vec<f64>>();
        let inner = TensorData::from_op(data, self.shape(), vec![self], Op::Pow(n));
        Self::new(inner)
    }

    /// Backward
    ///
    /// Computes the gradients of all the tensors that have been interacting and
    /// have `requires_grad` set to `true`.
    pub fn backward(&self) {
        let end_grad = self.0.borrow()._prev[0]
            .item()
            .iter()
            .map(|a| a * 2.0)
            .collect::<Vec<f64>>();
        self.add_to_grad(end_grad);
        self._backward()
    }

    /// Backward private
    ///
    /// Evokes Backward function in all components of the computational graph
    fn _backward(&self) {
        let t = self.0.borrow();
        if t.grad.is_some() && t._op.is_some() {
            t._op.as_ref().unwrap().backward(&t);
            if !t._prev.is_empty() {
                for prev in t._prev.clone() {
                    prev._backward()
                }
            }
        }
    }

    /// Multicast operation
    ///
    /// It ensures that the lower dimensions of the two tensors are the same if
    /// they are, then it performs the given operation elementwise, using the
    /// lower dimensional tensor as a convolution window.
    ///
    /// Accepts:
    /// * gt: data vector of greater `Tensor`
    /// * lt: data vector of less `Tensor`
    /// * gt_shape: shape of the greater `Tensor`
    /// * lt_shape: shape of the less `Tensor`
    /// * op: operation `Op`, permitted operations are `Add` and `Mul`
    fn multicast_op(gt: &Tensor, lt: &Tensor, op: Op) -> Vec<f64> {
        let gt_shape = gt.shape();
        let lt_shape = lt.shape();
        // ensure that th eoperation is within the permitted ones
        let pass = (op == Op::Add || op == Op::Sub || op == Op::Mul)
            && (gt_shape.len() > lt_shape.len() || gt_shape.len() == lt_shape.len());
        assert_eq!(true, pass);

        // remove the batch (1) dimensions
        let mut lt_shape = lt_shape.clone();
        for i in 0..lt_shape.len() {
            if lt_shape[i] == 1 && i < lt_shape.len() - 2 {
                lt_shape.remove(i);
            }
        }

        // ensure that the lower dimensions of the two tensors are the same
        let mut same_shape: Vec<usize> = gt_shape.clone();
        for i in 0..gt_shape.len() - lt_shape.len() {
            same_shape.remove(i);
        }
        assert_eq!(
            same_shape, lt_shape,
            "The lower dimensions of the two tensors are not equal"
        );

        let mut res = Vec::with_capacity(gt.length());
        let conv = lt.length();
        match op {
            Op::Add => {
                for i in (0..gt.length()).step_by(conv) {
                    let mut tmp = gt.0.borrow_mut().data[i..i + conv]
                        .iter()
                        .zip(lt.0.borrow().data.clone())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<f64>>();
                    res.append(&mut tmp);
                }
            }
            Op::Sub => {
                for i in (0..gt.length()).step_by(conv) {
                    let mut tmp = gt.0.borrow_mut().data[i..i + conv]
                        .iter()
                        .zip(lt.0.borrow().data.clone())
                        .map(|(a, b)| a - b)
                        .collect::<Vec<f64>>();
                    res.append(&mut tmp);
                }
            }
            Op::Mul => {
                for i in (0..gt.length()).step_by(conv) {
                    let mut tmp = gt.0.borrow_mut().data[i..i + conv]
                        .iter()
                        .zip(lt.0.borrow().data.clone())
                        .map(|(a, b)| a * b)
                        .collect::<Vec<f64>>();
                    res.append(&mut tmp);
                }
            }
            _ => unreachable!(),
        }
        res
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        // get lengths
        let self_len = self.length();
        let rhs_len = rhs.length();
        // get shapes
        let self_shape = self.shape();
        let rhs_shape = rhs.shape();

        let mut res = self.item();
        let mut res_shape = self.shape();
        let op = Op::Add;

        if rhs_len == 1 {
            for i in 0..self.length() {
                res[i] += rhs.0.borrow_mut().data[0];
            }
        } else {
            let (gt, gt_shape, lt, _lt_shape) = if self_len > rhs_len {
                (&self, self_shape, &rhs, rhs_shape)
            } else {
                (&rhs, rhs_shape, &self, self_shape)
            };
            res_shape = gt_shape;
            res = Self::multicast_op(gt, lt, op.clone());
        }
        let inner = TensorData::from_op(res, res_shape, vec![self, rhs], op);
        Self::new(inner)
    }
}

impl Add<i64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: i64) -> Self::Output {
        for i in 0..self.length() {
            self.0.borrow_mut().data[i] += rhs as f64;
        }
        self
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        for i in 0..self.length() {
            self.0.borrow_mut().data[i] += rhs;
        }
        self
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        // get lengths
        let self_len = self.length();
        let rhs_len = rhs.length();
        // get shapes
        let self_shape = self.shape();
        let rhs_shape = rhs.shape();

        let mut res = self.item();
        let mut res_shape = self.shape();
        let op = Op::Mul;

        if rhs_len == 1 {
            for i in 0..self.length() {
                res[i] *= rhs.0.borrow_mut().data[0];
            }
        } else {
            let (gt, gt_shape, lt, _lt_shape) = if self_len > rhs_len {
                (&self, self_shape, &rhs, rhs_shape)
            } else {
                (&rhs, rhs_shape, &self, self_shape)
            };
            res_shape = gt_shape;
            res = Self::multicast_op(gt, lt, op.clone());
        }
        let inner = TensorData::from_op(res, res_shape, vec![self, rhs], op);
        Self::new(inner)
    }
}

impl Mul<i64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: i64) -> Self::Output {
        for i in 0..self.length() {
            self.0.borrow_mut().data[i] *= rhs as f64;
        }
        self
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        for i in 0..self.length() {
            self.0.borrow_mut().data[i] *= rhs;
        }
        self
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        // get lengths
        let self_len = self.length();
        let rhs_len = rhs.length();
        // get shapes
        let self_shape = self.shape();
        let rhs_shape = rhs.shape();

        let mut res = self.item();
        let mut res_shape = self.shape();
        let op = Op::Sub;

        if rhs_len == 1 {
            for i in 0..self.length() {
                res[i] -= rhs.0.borrow_mut().data[0];
            }
        } else {
            let (gt, gt_shape, lt, _lt_shape) = if self_len > rhs_len {
                (&self, self_shape, &rhs, rhs_shape)
            } else {
                (&rhs, rhs_shape, &self, self_shape)
            };
            res_shape = gt_shape;
            res = Self::multicast_op(gt, lt, op.clone());
        }
        let inner = TensorData::from_op(res, res_shape, vec![self, rhs], op);
        Self::new(inner)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = String::new();
        let shape = self.shape();
        let data = self.tensor_to_str(data, 0, 0..shape.iter().product());
        let shape_str = format!("({shape:?})").replace("[", "").replace("]", "");
        let res = format!("Tensor({data}, shape: {shape_str}, dtype: f64)");
        write!(f, "{res}")
    }
}

// MACROS

#[macro_export]
macro_rules! randn {
    ($($element:expr),+) => {{
        use rand::Rng;
        // get shape
        let mut shape = Vec::new();
        // fill the shape
        $(shape.push($element);)*;
        // pass the shape to the `randn` method
        Tensor::randn(shape)
    }};
    ($($element:expr,)*) => {{
        crate::tensor::randn![$($element),*]
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! count_shape {
    (@COUNT; $($element:expr),*) => {
        <[()]>::len(&[$($crate::count_shape![@SUBST; $element]),*])
    };
    (@SUBST; $_element:expr) => { () };
}
