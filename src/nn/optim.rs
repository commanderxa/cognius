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
                .map(|(a, b)| b - a * self.lr)
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
