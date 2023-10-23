#[cfg(test)]
mod tests {
    use cognius::{
        data::{dataloader::Dataloader, dataset::Dataset, sample::Sample},
        Tensor,
    };

    struct MyDataset {
        data: Vec<Sample>,
    }

    impl MyDataset {
        fn new() -> Self {
            Self {
                data: vec![
                    (
                        Tensor::from_f64(&[0.0], &[1]),
                        Tensor::from_f64(&[0.0], &[1]),
                    ),
                    (
                        Tensor::from_f64(&[1.0], &[1]),
                        Tensor::from_f64(&[2.0], &[1]),
                    ),
                    (
                        Tensor::from_f64(&[2.0], &[1]),
                        Tensor::from_f64(&[4.0], &[1]),
                    ),
                    (
                        Tensor::from_f64(&[3.0], &[1]),
                        Tensor::from_f64(&[6.0], &[1]),
                    ),
                    (
                        Tensor::from_f64(&[4.0], &[1]),
                        Tensor::from_f64(&[8.0], &[1]),
                    ),
                    (
                        Tensor::from_f64(&[5.0], &[1]),
                        Tensor::from_f64(&[10.0], &[1]),
                    ),
                ],
            }
        }
    }

    impl Dataset<Sample> for MyDataset {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn sample(&self, index: usize) -> Sample {
            self.data[index].clone()
        }
    }

    #[test]
    fn dataset() {
        let _ = MyDataset::new();
    }

    #[test]
    fn dataloader() {
        let dataset = MyDataset::new();
        let dataloader = Dataloader::new(Box::new(dataset), 1, true);
        for (x, y) in dataloader.clone() {
            assert_eq!((x * 2).item(), y.item());
        }
    }
}
