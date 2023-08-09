use crate::Tensor;

pub struct SGD {
    parameters: Vec<Tensor>,
    lr: f64,
}

impl SGD {
    pub fn new(parameters: Vec<Tensor>, lr: f64) -> Self {
        Self { parameters, lr }
    }

    pub fn step(&self) {
        for i in 0..self.parameters.len() {
            let data: Vec<f64> = self.parameters[i]
                .grad()
                .unwrap()
                .iter()
                .zip(self.parameters[i].item())
                .map(|(a, b)| {
                    // w_i = w_(i-1) - lr * grad
                    // b = w(i-1)
                    // a = grad
                    b - self.lr * a
                })
                .collect();
            self.parameters[i].set_data(data);
        }
    }

    pub fn zero_grad(&self) {
        for i in 0..self.parameters.len() {
            self.parameters[i].0.borrow_mut().zero_grad();
        }
    }
}
