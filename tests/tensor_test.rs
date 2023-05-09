#[cfg(test)]
mod tests {
    use minigrad::Tensor;

    #[test]
    /// Valid shape of the tensor
    fn valid_shape() {
        Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![3, 2]);
        Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![1, 6]);
        Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![6, 1]);
    }

    #[test]
    #[should_panic]
    /// Invalid shape of the tensor
    fn invalid_shape() {
        Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![1, 3]);
    }

    #[test]
    fn zeros_like() {
        let a = Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        let a1 = Tensor::zeros_like(a.clone());
        assert_eq!(0.0, a1.item().iter().sum(), "zeros_like produces not zeros");
        assert_eq!(a.shape(), a1.shape(), "zeros_like produce wrong shape of a tensor");
        let b = Tensor::randn(vec![4, 10, 8]);
        let b1 = Tensor::zeros_like(b.clone());
        assert_eq!(0.0, b1.item().iter().sum(), "zeros_like produces not zeros");
        assert_eq!(b.shape(), b1.shape(), "zeros_like produce wrong shape of a tensor");
    }

    #[test]
    fn ones_like() {
        let a = Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        let a1 = Tensor::ones_like(a.clone());
        assert_eq!(1.0, a1.item().iter().product(), "ones_like produces not ones");
        assert_eq!(a.shape(), a1.shape(), "ones_like produce wrong shape of a tensor");
        let b = Tensor::randn(vec![4, 10, 8]);
        let b1 = Tensor::ones_like(b.clone());
        assert_eq!(1.0, b1.item().iter().product(), "ones_like produces not ones");
        assert_eq!(b.shape(), b1.shape(), "ones_like produce wrong shape of a tensor");
    }

    #[test]
    /// Matrix transpose
    fn t_2d() {
        let a = Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        let t = Tensor::from_f64(vec![0., 3., 1., 4., 2., 5.], vec![3, 2]);
        assert_eq!(a.t().shape(), t.shape(), "Shapes are wrong");
        assert_eq!(a.t().item(), t.item(), "Data is wrong");
    }

    #[test]
    /// Matrix multiplication
    fn mm_2d() {
        let a = Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        let b: Tensor = Tensor::from_f64(vec![6., 7., 8., 9., 10., 11.], vec![3, 2]);
        let c = Tensor::from_f64(vec![28., 31., 100., 112.], vec![2, 2]);
        let mm = Tensor::mm(a, b);
        assert_eq!(mm.item(), c.item());
        assert_eq!(mm.shape(), c.shape());
    }

    #[test]
    #[should_panic]
    /// Matrix multiplication
    fn mm_2d_panic() {
        let a: Tensor = Tensor::from_f64(vec![6., 7., 8., 9., 10., 11.], vec![3, 2]);
        let b = Tensor::from_f64(vec![28., 31., 100., 112.], vec![2, 2]);
        Tensor::mm(b, a);
    }

    #[test]
    /// New tensor of ordered numbers
    fn arange() {
        let a = Tensor::arange(0., 6., 1.);
        assert_eq!(a.shape(), vec![a.length()], "Shape is wrong");
        assert_eq!(a.item(), vec![0., 1., 2., 3., 4., 5.], "Data is wrong");
    }

    #[test]
    /// Reshape the tensor
    fn reshape() {
        let a = Tensor::from_f64(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![3, 4],
        );
        assert_eq!(a.shape(), vec![3, 4]);
        a.reshape(vec![1, 12]);
        assert_eq!(a.shape(), vec![1, 12]);
        a.reshape(vec![12, 1]);
        assert_eq!(a.shape(), vec![12, 1]);
        a.reshape(vec![2, 6]);
        assert_eq!(a.shape(), vec![2, 6]);
        a.reshape(vec![1, 3, 4]);
        assert_eq!(a.shape(), vec![1, 3, 4]);
        a.reshape(vec![2, 2, 3]);
        assert_eq!(a.shape(), vec![2, 2, 3]);
    }

    #[test]
    /// View the tensor
    fn view() {
        let a = Tensor::from_f64(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![3, 4],
        );
        let b = a.view(vec![1, 12]);
        assert_eq!(b.shape(), vec![1, 12]);
        let c = a.view(vec![12, 1]);
        assert_eq!(c.shape(), vec![12, 1]);
        let d = a.view(vec![2, 6]);
        assert_eq!(d.shape(), vec![2, 6]);
        let e = a.view(vec![1, 3, 4]);
        assert_eq!(e.shape(), vec![1, 3, 4]);
        let f = a.view(vec![2, 2, 3]);
        assert_eq!(f.shape(), vec![2, 2, 3]);
    }

    #[test]
    #[should_panic]
    /// Invalid view of the tensor
    fn view_invalid() {
        let a = Tensor::from_f64(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![3, 4],
        );
        a.reshape(vec![4, 5]);
    }

    #[test]
    fn pow() {
        let a = Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        let b = Tensor::from_f64(vec![0., 1., 4., 9., 16., 25.], vec![2, 3]);
        assert_eq!(a.pow(2).item(), b.item(), "Pow is wrong");
    }
}
