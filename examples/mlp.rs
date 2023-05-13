use minigrad::{nn::{Linear, Module, sigmoid}, Tensor};

fn main() {
    let mlp = MLP::new([4, 10, 20, 5, 1]);
    let x = Tensor::randn(vec![10, 4]);
    let out = mlp.forward(x);
    println!("out: {:?}", out.item());
    println!("shape: {:?}", out.shape());
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
}

