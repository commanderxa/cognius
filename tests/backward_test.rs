#[cfg(test)]
mod tests {
    use minigrad::{
        nn::{optim::SGD, sigmoid, Linear, Module},
        Tensor,
    };

    #[test]
    fn zero_grad() {
        let mlp = MLP::new([4, 10, 20, 5, 1]);
        let optim = SGD::new(mlp.parameters(), 1e-3);
        let x = Tensor::randn(vec![10, 4]);

        let out = mlp.forward(x.clone());

        out.backward();
        optim.step();
        optim.zero_grad();

        for item in mlp.parameters() {
            assert_eq!(item.grad().unwrap().iter().sum::<f64>(), 0.0);
        }
    }

    #[ignore]
    #[test]
    fn optim_step() {
        let mlp = MLP::new([4, 10, 20, 5, 1]);
        let optim = SGD::new(mlp.parameters(), 1e-3);
        let x = Tensor::randn(vec![10, 4]);

        let out = mlp.forward(x.clone());

        optim.zero_grad();
        out.backward();

        let grad_old = mlp.parameters().clone();

        optim.step();

        for (item, old) in mlp.parameters().iter().zip(grad_old) {
            assert_ne!(
                item.grad().unwrap().iter().sum::<f64>(),
                old.grad().unwrap().iter().sum::<f64>()
            );
        }
    }

    struct MLP {
        linear1: Linear,
        linear2: Linear,
        linear3: Linear,
        linear4: Linear,
    }

    impl MLP {
        pub fn new(features: [usize; 5]) -> Self {
            Self {
                linear1: Linear::new(features[0], features[1]),
                linear2: Linear::new(features[1], features[2]),
                linear3: Linear::new(features[2], features[3]),
                linear4: Linear::new(features[3], features[4]),
            }
        }
    }

    impl Module for MLP {
        fn module_name(&self) -> String {
            "MLP".to_owned()
        }

        fn forward(&self, x: Tensor) -> Tensor {
            let x = self.linear1.forward(x);
            let x = self.linear2.forward(x);
            let x = self.linear3.forward(x);
            let x = self.linear4.forward(x);
            sigmoid(x)
        }

        fn parameters(&self) -> Vec<Tensor> {
            let mut parameters = self.linear1.parameters();
            parameters.append(&mut self.linear2.parameters());
            parameters.append(&mut self.linear3.parameters());
            parameters.append(&mut self.linear4.parameters());
            parameters
        }
    }
}
