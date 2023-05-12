use minigrad::{nn::{Linear, Module, sigmoid}, Tensor};

fn main() {
    let linear = Linear::new(20, 10);
    let x = Tensor::randn(vec![2, 20]);
    println!("Weights: {}", linear.weight);
    println!("IN:\n{x}");
    let out = linear.forward(x);
    println!("OUT:\n{out}");
    let out = sigmoid(out);
    println!("OUT:\n{out}");
}