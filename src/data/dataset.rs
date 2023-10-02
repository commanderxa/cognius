pub trait Dataset<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn sample(&self, index: usize) -> T;
}
