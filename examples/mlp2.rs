use cognius::{
    nn::{optim::SGD, Linear, MSELoss, Module},
    Tensor,
};

fn main() {
    let epochs = 10;
    let criterion = MSELoss::new();
    let mlp = MLP::new([1, 1]);
    let optim = SGD::new(mlp.parameters(), 1e-1);

    let data = vec![
        Tensor::from_f64(vec![1.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0], vec![1, 1]),
        Tensor::from_f64(vec![3.0], vec![1, 1]),
        Tensor::from_f64(vec![4.0], vec![1, 1]),
        Tensor::from_f64(vec![5.0], vec![1, 1]),
        Tensor::from_f64(vec![6.0], vec![1, 1]),
        Tensor::from_f64(vec![7.0], vec![1, 1]),
        Tensor::from_f64(vec![8.0], vec![1, 1]),
        Tensor::from_f64(vec![9.0], vec![1, 1]),
        Tensor::from_f64(vec![10.0], vec![1, 1]),
    ];
    let targets = vec![
        Tensor::from_f64(vec![2.0 * 1.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 2.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 3.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 4.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 5.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 6.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 7.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 8.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 9.0], vec![1, 1]),
        Tensor::from_f64(vec![2.0 * 10.0], vec![1, 1]),
    ];

    let test_data = vec![
        Tensor::from_f64(vec![11.0], vec![1, 1]),
        Tensor::from_f64(vec![-2.0], vec![1, 1]),
    ];
    let test_targets = vec![
        Tensor::from_f64(vec![22.0], vec![1, 1]),
        Tensor::from_f64(vec![-4.0], vec![1, 1]),
    ];

    for epoch in 1..=epochs {
        let mut losses = 0.0;
        for (x, y) in data.iter().zip(targets.clone()) {
            optim.zero_grad();

            let x = x.clone();
            let y = y.clone();
            let out = mlp.forward(x.clone());

            println!("IN: {0}\tOUT: {1:?}", x, out);
            let loss = criterion.measure(out, y);

            loss.backward();
            println!("\nLOSS: {:?}\n", loss);
            println!("MODEL INFO: {:?}", mlp.parameters());
            optim.step();

            losses += loss.item()[0];
        }
        println!(
            "Epoch: {epoch}/{epochs} | loss: {:.10}",
            (losses / (epochs as f64 * 2.0))
        );
    }

    for (x, y) in data.iter().zip(targets.clone()) {
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
        for (i, param) in mlp.parameters().iter().enumerate() {
            let sum = param.grad().unwrap().iter().sum::<f64>();
            println!("{i} ==> {sum:.50}");
        }
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
        for (i, param) in mlp.parameters().iter().enumerate() {
            let sum = param.grad().unwrap().iter().sum::<f64>();
            println!("{i} ==> {sum:.50}");
        }
    }

    println!("MODEL: {:?}", mlp.parameters());
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
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let parameters = self.linear1.parameters();
        // parameters.append(&mut self.linear2.parameters());
        // parameters.append(&mut self.linear3.parameters());
        // parameters.append(&mut self.linear4.parameters());
        parameters
    }
}
