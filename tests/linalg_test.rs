#[cfg(test)]
mod tests {
    use cognius::{linalg, Tensor};

    #[test]
    /// Matrix multiplication
    fn matmul_2d() {
        let a = Tensor::from_f64(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]);
        let b: Tensor = Tensor::from_f64(vec![6., 7., 8., 9., 10., 11.], vec![3, 2]);
        let c = Tensor::from_f64(vec![28., 31., 100., 112.], vec![2, 2]);
        let mm = linalg::matmul(a, b);
        assert_eq!(mm.item(), c.item());
        assert_eq!(mm.shape, c.shape);
    }

    #[test]
    #[should_panic]
    /// Matrix multiplication
    fn matmul_2d_panic() {
        let a: Tensor = Tensor::from_f64(vec![6., 7., 8., 9., 10., 11.], vec![3, 2]);
        let b = Tensor::from_f64(vec![28., 31., 100., 112.], vec![2, 2]);
        linalg::matmul(b, a);
    }

    #[test]
    /// Batched matrix multiplication
    fn matmul_batched() {
        let a = Tensor::from_f64(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
            ],
            vec![2, 4, 5],
        );
        let b = Tensor::from_f64(
            vec![9.0, 5.0, 3.0, 2.0, 6.0, 9.0, 5.0, 3.0, 2.0, 6.0],
            vec![5, 2],
        );
        let right = Tensor::from_f64(
            vec![
                63.0000, 78.0000, 188.0000, 203.0000, 313.0000, 328.0000, 438.0000, 453.0000,
                563.0000, 578.0000, 688.0000, 703.0000, 813.0000, 828.0000, 938.0000, 953.0000,
            ],
            vec![2, 4, 2],
        );
        let c = linalg::matmul(a.clone(), b.clone());
        assert_eq!(c.item(), right.item());
        assert_eq!(c.shape, right.shape);
    }

    #[test]
    fn cross_1d() {
        let a = Tensor::from_f64(vec![0.6, -20.5, 5.8], vec![3]);
        let b = Tensor::from_f64(vec![10.2, -4.6, -34.], vec![3]);
        let c = linalg::cross(a.clone(), b.clone());
        assert_eq!(vec![723.6800, 79.5600, 206.3400], c.item());
    }

    #[test]
    fn mul_scalar_2d() {
        let a = Tensor::arange(1.0, 11.0, 1.0);
        let a = a.reshape(vec![2, 5]);
        let b = 5;
        let c = a * b;
        assert_eq!(
            c.storage(),
            vec![
                5.0 * 1.0,
                5.0 * 2.0,
                5.0 * 3.0,
                5.0 * 4.0,
                5.0 * 5.0,
                5.0 * 6.0,
                5.0 * 7.0,
                5.0 * 8.0,
                5.0 * 9.0,
                5.0 * 10.0,
            ]
        )
    }

    #[test]
    fn mul_scalar_3d() {
        let a = Tensor::arange(1.0, 9.0, 1.0);
        let a = a.reshape(vec![2, 2, 2]);
        let b = 5;
        let c = a * b;
        assert_eq!(
            c.storage(),
            vec![
                5.0 * 1.0,
                5.0 * 2.0,
                5.0 * 3.0,
                5.0 * 4.0,
                5.0 * 5.0,
                5.0 * 6.0,
                5.0 * 7.0,
                5.0 * 8.0,
            ]
        )
    }

    #[test]
    fn mul_3d_by_1d() {
        let a = Tensor::arange(1.0, 9.0, 1.0).reshape(vec![2, 2, 2]);
        let b = Tensor::from_f64(vec![-2.0, 2.0], vec![2]);
        let c = a * b;
        assert_eq!(
            c.storage(),
            vec![-2.0, 4.0, -6.0, 8.0, -10.0, 12.0, -14.0, 16.0]
        );
    }

    #[test]
    fn mul_3d_by_2d() {
        let a = Tensor::arange(1.0, 9.0, 1.0).reshape(vec![2, 2, 2]);
        let b = Tensor::from_f64(vec![-2.0, 3.0, 10.0, 4.0], vec![2, 2]);
        let c = a * b;
        assert_eq!(
            c.storage(),
            vec![-2.0, 6.0, 30.0, 16.0, -10.0, 18.0, 70.0, 32.0]
        );
    }

    #[test]
    fn add_scalar_3d() {
        let a = Tensor::arange(1.0, 9.0, 1.0).reshape(vec![2, 2, 2]);
        let b = 2.0;
        let c = a + b;
        assert_eq!(c.storage(), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn add_elementwise_3d() {
        let a = Tensor::arange(1.0, 9.0, 1.0).reshape(vec![2, 2, 2]);
        let b = Tensor::arange(8.0, 0.0, -1.0).reshape(vec![2, 2, 2]);
        let c = a + b;
        assert_eq!(c.storage(), vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn add_3d_by_2d() {
        let a = Tensor::arange(1.0, 9.0, 1.0).reshape(vec![2, 2, 2]);
        let b = Tensor::from_f64(vec![-2.0, 3.0, 10.0, 4.0], vec![2, 2]);
        let c = a + b;
        assert_eq!(
            c.storage(),
            vec![-1.0, 5.0, 13.0, 8.0, 3.0, 9.0, 17.0, 12.0]
        );
    }
}
