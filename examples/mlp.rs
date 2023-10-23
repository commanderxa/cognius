use cognius::{
    nn::{optim::SGD, sigmoid, Linear, MSELoss, Module},
    Tensor,
};

fn main() {
    let epochs = 10;
    let criterion = MSELoss::new();
    let mlp = MLP::new([2, 1]);
    let optim = SGD::new(mlp.parameters(), 1.0);

    let data = vec![
        Tensor::from_f64(&[9., 3.0], &[1, 2]),
        Tensor::from_f64(&[2., 3.0], &[1, 2]),
        Tensor::from_f64(&[3., 7.0], &[1, 2]),
        Tensor::from_f64(&[8., 6.0], &[1, 2]),
        Tensor::from_f64(&[4., 4.0], &[1, 2]),
        Tensor::from_f64(&[4., 1.0], &[1, 2]),
        Tensor::from_f64(&[5., 2.0], &[1, 2]),
        Tensor::from_f64(&[2., 5.0], &[1, 2]),
        Tensor::from_f64(&[5., 6.0], &[1, 2]),
        Tensor::from_f64(&[7., 2.0], &[1, 2]),
        Tensor::from_f64(&[9., 1.0], &[1, 2]),
    ];
    let targets = vec![
        Tensor::from_f64(&[0.0], &[1, 1]),
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[0.0], &[1, 1]),
        Tensor::from_f64(&[0.0], &[1, 1]),
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[0.0], &[1, 1]),
        Tensor::from_f64(&[0.0], &[1, 1]),
    ];

    let test_data = vec![
        Tensor::from_f64(&[6., 6.0], &[1, 2]),
        Tensor::from_f64(&[8., 2.0], &[1, 2]),
    ];
    let test_targets = vec![
        Tensor::from_f64(&[1.0], &[1, 1]),
        Tensor::from_f64(&[0.0], &[1, 1]),
    ];

    for epoch in 1..=epochs {
        let mut losses = 0.0;
        for (x, y) in data.iter().zip(targets.clone()) {
            optim.zero_grad();

            let x = x.clone();
            let y = y.clone();
            let out = mlp.forward(x);

            let loss = criterion.measure(out, y);

            loss.backward();
            println!("BACKWARD:\n{:?}", mlp.parameters());
            optim.step();

            losses += loss.item()[0];
        }
        println!("Epoch: {epoch}/{epochs} | loss: {:.10}", losses);
    }

    for (x, y) in test_data.iter().zip(test_targets.clone()) {
        let out = mlp.forward(x.clone());

        let loss = criterion.measure(out.clone(), y.clone());
        println!(
            "\nDATA: {0}\nTARGETS: {1}\nOUT:  {2}\nLOSS: {3:?}",
            x,
            y.item()[0],
            out.item()[0],
            loss.item()[0]
        );
        loss.backward();
    }
}

struct MLP {
    linear1: Linear,
    // linear2: Linear,
    // linear3: Linear,
    // linear4: Linear,
}

impl MLP {
    pub fn new(features: [usize; 2]) -> Self {
        Self {
            linear1: Linear::new(features[0], features[1]),
            // linear2: Linear::new(features[1], features[2]),
            // linear3: Linear::new(features[2], features[3]),
            // linear4: Linear::new(features[3], features[4]),
        }
    }
}

impl Module for MLP {
    fn module_name(&self) -> String {
        "MLP".to_owned()
    }

    fn forward(&self, x: Tensor) -> Tensor {
        let x = self.linear1.forward(x);
        // let x = self.linear2.forward(x);
        // let x = self.linear3.forward(x);
        // let x = self.linear4.forward(x);
        sigmoid(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let parameters = self.linear1.parameters();
        // parameters.append(&mut self.linear2.parameters());
        // parameters.append(&mut self.linear3.parameters());
        // parameters.append(&mut self.linear4.parameters());
        parameters
    }
}
