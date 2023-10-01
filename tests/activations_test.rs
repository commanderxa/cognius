#[cfg(test)]
mod tests {
    use cognius::{nn, Tensor};

    #[test]
    fn sigmoid() {
        let t = Tensor::from_f64(vec![0.0], vec![1]);
        let t_act = nn::sigmoid(t);
        assert_eq!(t_act.item()[0], 0.5);
    }

    #[test]
    fn relu() {
        let t = Tensor::from_f64(vec![0.0], vec![1]);
        let t_act = nn::relu(t);
        assert_eq!(t_act.item()[0], 0.0);

        let t = Tensor::from_f64(vec![-20.0], vec![1]);
        let t_act = nn::relu(t);
        assert_eq!(t_act.item()[0], 0.0);

        let t = Tensor::from_f64(vec![100.0], vec![1]);
        let t_act = nn::relu(t);
        assert_eq!(t_act.item()[0], 100.0);
    }
}
