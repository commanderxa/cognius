#[cfg(test)]
mod tests {
    use minigrad::{Tensor, nn::relu};


    #[test]
    fn simple() {
        let t1 = Tensor::from_f64(vec![2.0, 3.0, 1.0, 5.0, 6.0, 1.0], vec![2, 3]);
        let t2 = Tensor::ones(vec![2, 3]);
        let pr1 = t1.clone() + t2.clone();
        let pr2 = relu(pr1);
        pr2.backward();

        print!("{pr2}");
    }
}
