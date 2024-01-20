#[cfg(test)]
mod tests {
    use cognius::{nn::functional as F, Tensor};

    #[test]
    fn sigmoid() {
        let t = Tensor::from_f64(&[0.0], &[1]);
        let t_act = F::sigmoid(t);
        assert_eq!(t_act.item()[0], 0.5);
    }

    #[test]
    fn relu() {
        let t = Tensor::from_f64(&[0.0], &[1]);
        let t_act = F::relu(t);
        assert_eq!(t_act.item()[0], 0.0);

        let t = Tensor::from_f64(&[-20.0], &[1]);
        let t_act = F::relu(t);
        assert_eq!(t_act.item()[0], 0.0);

        let t = Tensor::from_f64(&[100.0], &[1]);
        let t_act = F::relu(t);
        assert_eq!(t_act.item()[0], 100.0);
    }
}
