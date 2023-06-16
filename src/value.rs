use std::{
    cell::{Ref, RefCell},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc, f64::consts::E,
};

use crate::op::Op;

#[derive(Clone)]
/// Value
/// 
/// The base unit of this crate.
/// 
/// It represents the reference to the `InnerValue` with some properties:
/// * Allows to mutate the value inside
/// * Allows to create multiple instance of the same `Value`
/// 
/// To create multiple instances of the same `Value` call `clone()` method
/// on the `Value` that you need. As the `Value` is a reference to the 
/// data inside, `clone()` will not make a copy of the data, but the copy 
/// of a reference, therefore the actual data will be mutated as intended.
pub struct Value(Rc<RefCell<InnerValue>>);

impl Value {
    /// Creates a new instance of `Value`.
    /// 
    /// Accepts:
    /// * value: `InnerValue`
    pub fn new(value: InnerValue) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }

    /// Creates a new instance of `Value` from data.
    /// 
    /// Accepts:
    /// * data: `f64`
    /// 
    /// If you don't have an `InnerValue`, then use this method.\
    /// Use of this method is preferred as it simplifies the code.
    pub fn from(data: f64) -> Self {
        Self(Rc::new(RefCell::new(InnerValue::new(data))))
    }

    /// Creates a new instance of `Value` from a `Vec<f64>`.
    /// 
    /// Accepts:
    /// * data: Vector of `f64`
    /// 
    /// This method will convert a vector of `f64` values into the
    /// vector of `Values`.
    pub fn from_vec(data: Vec<f64>) -> Vec<Self> {
        let mut result = Vec::with_capacity(data.len());
        for item in data {
            result.push(Value::from(item));
        }
        result
    }

    /// Returns the `f64` data inside the `InnerValue`.
    pub fn item(&self) -> f64 {
        self.0.borrow().data
    }

    /// Add a `f64` value to the data inside the `InnerValue`.
    pub fn add_data(&self, data: f64) {
        self.0.borrow_mut().data += data;
    }

    /// Returns the gradient inside the `InnerValue`.
    pub fn get_grad(&self) -> f64 {
        self.0.borrow().grad
    }

    /// Add a `f64` value to the gradient inside the `InnerValue`.
    pub fn add_grad(&self, data: f64) {
        self.0.borrow_mut().grad += data;
    }

    /// Sets the gradient inside the `InnerValue` to some `f64` value.
    /// 
    /// Accepts:
    /// * data: `f64`
    pub fn set_grad(&self, data: f64) {
        self.0.borrow_mut().grad = data;
    }

    /// Backward
    /// 
    /// Initial `backward` method.
    /// 
    /// It sets the first gradient to be 1, then it calls the `_backward` method. 
    /// See the documentation for `_backward`.
    pub fn backward(&self) {
        // set self.grad to 1 as it's an inital backprop point
        self.add_grad(1.0);
        let value = self.0.borrow();

        if let Some(backward) = value._backward {
            backward(&value);
            if let Some(left) = value._prev.get(0) {
                left._backward();
            }
            if let Some(right) = value._prev.get(1) {
                right._backward();
            }
        }
    }

    /// _backward
    /// 
    /// A private method created to be called from `backward` method.
    /// See the documentation for `backward`.
    /// 
    /// It calls the `backward` function (if) stored inside the value,
    /// then it rucersively calls itself, but for the previous nodes 
    /// of the value stored inside `_prev`.
    fn _backward(&self) {
        let value = self.0.borrow();

        // set self.grad to 1 as it's an inital backprop point
        if let Some(backward) = value._backward {
            backward(&value);
            if let Some(left) = value._prev.get(0) {
                left._backward();
            }
            if let Some(right) = value._prev.get(1) {
                right._backward();
            }
        }
    }

    /// Powers the value
    /// 
    /// Accepts `n` integer in which the value will be powered.
    /// 
    /// For backpropagation it stores the `n` inside the `Op::Pow(n)`.
    pub fn pow(self, n: i32) -> Value {
        let backward = |value: &Ref<InnerValue>| {
            let left = value._prev[0].item();
            if let Op::Pow(n) = value._op.as_ref().unwrap() {
                let n = n.clone();
                value._prev[0].add_grad(n as f64 * left.powi(n - 1) * value.grad);
            }
        };

        let data = self.item().powi(n);
        let set = vec![self];
        let inner = InnerValue::from_op(data, set, backward, Op::Pow(n));
        Value::new(inner)
    }

    /// ReLU implementation
    /// 
    /// This method behaves like ReLU:
    /// * Returns the value only if it is greater than 0
    /// * Returns 0 otherwise
    pub fn relu(self) -> Value {
        let backward = |value: &Ref<InnerValue>| {
            let grad = if value.data > 0.0 {
                1.0 * value.grad
            } else {
                0.0
            };

            value._prev[0].add_grad(grad);
        };

        let data = if self.item() > 0.0 {
            self.item()
        } else {
            0.0
        };

        let set = vec![self];
        let inner = InnerValue::from_op(data, set, backward, Op::ReLU);
        Value::new(inner)
    }

    /// Sigmoid implementation
    pub fn sigmoid(self) -> Value {
        let backward = |value: &Ref<InnerValue>| {
            if let Op::Sigmoid(x) = value._op.clone().unwrap() {
                let exp = E.powf(x) + 1.0;
                let grad = E.powf(x) / exp.powi(2);
    
                value._prev[0].add_grad(grad);
            }
        };

        let x = self.item();
        let data = 1.0 / (1.0 + E.powf(x));
        let set = vec![self];
        let inner = InnerValue::from_op(data, set, backward, Op::Sigmoid(x));
        Value::new(inner)
    }
}

/// Inner Value
/// 
/// It represents the value itself with all its necessary fields.\
/// See the documentation for `Value`.
pub struct InnerValue {
    data: f64,
    grad: f64,
    _backward: Option<fn(value: &Ref<InnerValue>)>,
    _prev: Vec<Value>,
    _op: Option<Op>,
}

impl InnerValue {
    /// Creates a new instance of `InnerValue`.
    /// 
    /// Accepts:
    /// * data: `f64`
    pub fn new(data: f64) -> Self {
        Self {
            data: data,
            grad: 0.0,
            _backward: None,
            _prev: Vec::new(),
            _op: None,
        }
    }

    /// Creates an instance of `InnerValue` that is made with `Op`.
    /// 
    /// Accepts:
    /// * data: `f64`
    /// * prev: Vector of `Value`
    /// * backward: a function that takes a reference to the `InnerValue`
    /// * op: the `Op` enum
    pub fn from_op(data: f64, prev: Vec<Value>, backward: fn(&Ref<InnerValue>), op: Op) -> Self {
        Self {
            data: data,
            grad: 0.0,
            _backward: Some(backward),
            _prev: prev,
            _op: Some(op),
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let backward = |value: &Ref<InnerValue>| {
            // In addition each value gradient is: 1.0 * value.grad
            value._prev[0].add_grad(value.grad);
            value._prev[1].add_grad(value.grad);
        };

        let data = self.item() + rhs.item();
        let set = vec![self, rhs];
        let inner = InnerValue::from_op(data, set, backward, Op::Add);
        Value::new(inner)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let backward = |value: &Ref<InnerValue>| {
            // In multiplication each value gradient is: other.data * value.grad
            let left = value._prev[0].item();
            let right = value._prev[1].item();
            value._prev[0].add_grad(right * value.grad);
            value._prev[1].add_grad(left * value.grad);
        };

        let data = self.item() * rhs.item();
        let set = vec![self, rhs];
        let inner = InnerValue::from_op(data, set, backward, Op::Mul);
        Value::new(inner)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * Value::new(InnerValue::new(-1.0))
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value: {{ data: {0}, prev: {{ {1}, {2} }}, op: {3}, grad: {4} }}",
            self.item(),
            if let Some(left) = self.0.borrow()._prev.get(0) {
                format!("{}", left)
            } else {
                "None".to_string()
            },
            if let Some(right) = self.0.borrow()._prev.get(1) {
                format!("{}", right)
            } else {
                "None".to_string()
            },
            if let Some(op) = &self.0.borrow()._op {
                format!("{}", op)
            } else {
                "None".to_string()
            },
            self.0.borrow().grad,
        )
    }
}
