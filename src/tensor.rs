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
pub struct Tensor {
    pub(crate) inner: Rc<RefCell<TensorData>>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new instance of a `Tensor`.
    pub(crate) fn new(inner: TensorData) -> Self {
        let shape = inner.shape.clone();
        Self {
            inner: Rc::new(RefCell::new(inner)),
            shape: shape,
        }
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
    pub(crate) fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Returns the total length of the vector. It obtains the length by taking
    /// the product of the shape of the tensor.
    ///
    /// E.g. if the shape of the tensor is (3, 2, 3), then the length is
    /// (3 * 2 * 3) => 18.
    pub fn length(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns the data of `Tensor` as it is stored.
    pub fn storage(&self) -> Vec<f64> {
        self.inner.borrow().data.clone()
    }

    /// Computes the stides of a `Tensor`.
    ///
    /// These strides are used to perform certain operations on a `Tensor`,
    /// allowing to recompute the `Tensor` itself
    pub fn stride(&self) -> Vec<usize> {
        let shape = self.shape();
        let original_shape = self.inner.borrow().shape();
        let mut stride: Vec<usize> = vec![0; shape.len()];
        // set last stride dim to 1
        stride[shape.len() - 1] = 1;
        // track the last cumulative dimension (from the rear)
        let mut prev_original = stride[shape.len() - 1];
        let mut last_idx = shape.len() - 1;
        for i in (0..(shape.len())).rev() {
            if original_shape[i] == shape[i] {
                stride[i] = 1;
                prev_original = stride[i];
                last_idx = i;
                break;
            } else {
                stride[i] = 0;
            }
        }
        // compute stride
        for i in (0..last_idx).rev() {
            stride[i] = shape[i + 1] * prev_original;
            // set the copied dimension to 0
            if original_shape[i] != shape[i] {
                stride[i] = 0;
                continue;
            }
            // update the last cumulative dimensions
            prev_original = stride[i];
        }
        stride
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
        let storage = self.storage();
        let stride = self.stride();
        let shape = self.shape();
        let mut mask = vec![0; shape.len()];
        let mut data = vec![0.0; self.length()];
        // iterate over storage data
        for i in 0..self.length() {
            // compute index of past position of data
            data[i] = storage[stride.iter().zip(&mask).map(|(a, b)| a * b).sum::<usize>()];
            // iterate over shape
            for j in (0..shape.len()).rev() {
                // skip the properly filled dims
                if shape[j] - 1 == mask[j] {
                    continue;
                }
                // increment the necessary mask dim
                mask[j] += 1;
                // set to 0 all prevous shape dims
                for k in ((j + 1)..shape.len()).rev() {
                    mask[k] = 0;
                }
                break;
            }
        }
        data
    }

    /// Defines the `Tensor` behavior.
    ///
    /// The tensor property `requires_grad` is `true` by default, which means
    /// that the `Tensor` has a gradient, but this gradient might be sent to
    /// `None` if is not necessary.
    pub fn requires_grad(self, value: bool) -> Self {
        if value {
            self.inner.borrow_mut().grad = Some(vec![0.0; self.length()]);
        } else {
            self.inner.borrow_mut().grad = None;
        }
        self
    }

    /// Transpose
    ///
    /// This method transposes the tensor, it changes the shape.
    /// The rows become the columns and vice versa.
    ///
    /// E.g. if the shape was (2, 3) it will make (3, 2).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut t = self.clone();
        // transpose tensor of 2 and more dimensions
        if t.shape().len() >= 2 {
            t.shape.swap(dim0, dim1);
        }
        t
    }

    /// Expects input to be <= 2D and transposes 0 and 1 dims.
    ///
    /// 0D and 1D tensors are returned without any transpose performed
    pub fn t(&self) -> Self {
        self.transpose(0, 1)
    }

    /// Returns the tensor of the new shape.
    ///
    /// Tensor of shape (2, 3) might be viewed as (3, 2), (6, 1), (1, 6).
    /// Tensor can be viewed as any shape, only if the length of this shape is
    /// the same as the length of the previous shape.
    pub fn view(&self, shape: &[usize]) -> Self {
        assert_eq!(
            self.length(),
            shape.iter().product(),
            "Length of the new shape: {} does not match the length of the old one: {}",
            shape.iter().product::<usize>(),
            self.length()
        );
        let mut reshaped_t = self.clone();
        reshaped_t.shape = shape.to_vec();
        reshaped_t
    }

    /// Reshapes the tensor.
    ///
    /// Tensor of shape (2, 3) might be reshaped to (3, 2), (6, 1), (1, 6).
    /// Tensor can be reshaped into any shape, only if the length of this shape
    /// is the same as the length of the previous shape.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        assert_eq!(
            self.length(),
            shape.iter().product(),
            "Length of the new shape: {} does not match the length of the old one: {}",
            shape.iter().product::<usize>(),
            self.length()
        );
        let mut reshaped_t = self.clone();
        reshaped_t.shape = shape.to_vec();
        reshaped_t
    }

    /// Exponents each value of the `Tensor`.
    ///
    /// `exp(x)` => `e^(x)`.
    pub fn exp(&self) -> Tensor {
        let mut data = self.item();
        for item in data.iter_mut() {
            *item = E.powf(*item);
        }
        let inner = TensorData::from_op(
            data,
            self.shape(),
            vec![self.clone()],
            Op::Exp(self.clone()),
        );
        Tensor::new(inner)
    }

    /// Exapnds tesnor along its dimensions.
    ///
    /// Takes new shape.
    pub fn expand(&self, new_shape: Vec<usize>) -> Self {
        assert!(self.shape().len() <= new_shape.len(), "The number of sizes provided ({:?}) must be equal or greater than the number of sizes in the tensor ({:?})", self.shape().len(), new_shape.len());
        let mut _old_shape = self.shape();
        // check if batch dims have to be added in th front
        let dims_to_add = new_shape.len() - _old_shape.len();
        let mut old_shape: Vec<usize> = vec![];
        // push neccessary front batch dims
        for _ in 0..dims_to_add {
            old_shape.push(1);
        }
        // append the rest of the shape
        old_shape.append(&mut _old_shape);
        // check if sizes are consistent
        for i in (0..new_shape.len()).rev() {
            assert!(
                old_shape[i] == new_shape[i] || (new_shape[i] == 1 || old_shape[i] == 1),
                "The expanded size of the tensor ({}) must match the existing size ({}) at dimension ({})", 
                format!("{}", new_shape[i]), 
                format!("{}", old_shape[i]), 
                format!("{}", i)
            );
        }

        // change tensor properties
        let mut t = self.clone();
        t.shape = new_shape;
        t
    }

    /// Converts the tensor to a `String`, so that it can be printed.
    fn tensor_to_str(&self, tensor_str: String, level: usize, range: Range<usize>) -> String {
        let mut width = 1;
        for i in self.storage() {
            let s = (i.floor() as i64).to_string();
            if s.len() > width {
                width = s.len();
            }
        }
        width += 5;
        let result = self._tensor_to_str(tensor_str, level, range, width);
        result
    }

    fn _tensor_to_str(
        &self,
        tensor_str: String,
        level: usize,
        range: Range<usize>,
        width: usize,
    ) -> String {
        // the length of the range
        let len: usize = range.end - range.start;
        // the current dimension from the shape
        let dim = self.shape()[level];
        // convolution to iterate over the data
        let conv = len / dim;
        // the length of shape vector
        let shape_size = self.shape().len();
        let item = self.item();
        // denote the start of the dimension
        let mut result = String::from("[");
        // iterate over the dimension => print a vector
        for i in (range.start..range.end).step_by(conv) {
            // if the dimension is the last one
            let mut spaces: usize = 0;
            if shape_size - 1 == level {
                let s = (item[i].floor() as i64).to_string();
                if s.len() < width {
                    spaces = width - (s.len() + 5);
                }
                for _ in 0..spaces {
                    result.push_str(&" ");
                }
                let mut num = format!("{:.4}", item[i]);
                if i < self.shape()[level] - 1 {
                    num.push_str(", ");
                }
                result.push_str(num.as_str());
            }
            // if the iteration is over the last 2 dimensions => print a matrix
            else if shape_size - 2 == level {
                result.push('[');
                for j in 0..self.shape()[level + 1] {
                    let s = (item[i + j].floor() as i64).to_string();
                    if s.len() < width {
                        spaces = width - (s.len() + 5);
                    }
                    for _ in 0..spaces {
                        result.push_str(&" ");
                    }
                    let mut num = format!("{:.4}", item[i + j]);
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
                    result.push(']');
                }
            } else {
                // else, fall further into the next dimensions
                result.push_str(
                    self._tensor_to_str(tensor_str.clone(), level + 1, i..(i + conv), width)
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
        result.push(']');
        result
    }

    /// Add a `Vec<f64>` value to the gradient inside the `TensorData`.
    pub(crate) fn add_to_grad(&self, data: Vec<f64>) {
        let mut t = self.inner.borrow_mut();
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
        self.inner.borrow().grad.clone()
    }

    pub fn set_data(&self, data: Vec<f64>) {
        self.inner.borrow_mut().data = data;
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
        let end_grad = self.inner.borrow()._prev[0]
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
        let t = self.inner.borrow();
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
        assert!(pass);

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
                    let mut tmp = gt.inner.borrow_mut().data[i..i + conv]
                        .iter()
                        .zip(lt.inner.borrow().data.clone())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<f64>>();
                    res.append(&mut tmp);
                }
            }
            Op::Sub => {
                for i in (0..gt.length()).step_by(conv) {
                    let mut tmp = gt.inner.borrow_mut().data[i..i + conv]
                        .iter()
                        .zip(lt.inner.borrow().data.clone())
                        .map(|(a, b)| a - b)
                        .collect::<Vec<f64>>();
                    res.append(&mut tmp);
                }
            }
            Op::Mul => {
                for i in (0..gt.length()).step_by(conv) {
                    let mut tmp = gt.inner.borrow_mut().data[i..i + conv]
                        .iter()
                        .zip(lt.inner.borrow().data.clone())
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
            for item in res.iter_mut() {
                *item += rhs.inner.borrow_mut().data[0];
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
            self.inner.borrow_mut().data[i] += rhs as f64;
        }
        self
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        for i in 0..self.length() {
            self.inner.borrow_mut().data[i] += rhs;
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
            for item in res.iter_mut() {
                *item *= rhs.inner.borrow_mut().data[0];
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
            self.inner.borrow_mut().data[i] *= rhs as f64;
        }
        self
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        for i in 0..self.length() {
            self.inner.borrow_mut().data[i] *= rhs;
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
            for item in res.iter_mut() {
                *item -= rhs.inner.borrow_mut().data[0];
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
        let shape_str = format!("({shape:?})").replace('[', "").replace(']', "");
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
        $crate::tensor::randn![$($element),*]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Tests the original shape after expansion.
    /// It cannot be tested outside of this module as `Tensor`'s field `inner`
    /// is not accessible outside of this crate
    fn expand() {
        let a = Tensor::ones(vec![1, 1, 3, 1, 3, 3]);
        let a = a.expand(vec![2, 2, 3, 3, 3, 3]);
        assert_eq!(a.shape, vec![2, 2, 3, 3, 3, 3]);
        assert_eq!(a.inner.borrow().shape, vec![1, 1, 3, 1, 3, 3]);
    }
}
