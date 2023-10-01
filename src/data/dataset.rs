pub trait Dataset<T> {
    fn len(&self) -> usize;
    fn sample(&self, index: usize) -> T;
}
