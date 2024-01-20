/// # Dataset Trait
///
/// Defines the behavior of `Dataset`.
/// Methods:
/// - len - returns the length of the dataset. E.g. number of rows from a table
/// - sample - returns a sample given the index
pub trait Dataset<T> {
    /// Returns length of `Dataset`
    fn len(&self) -> usize;

    /// Provides a Sample of <T> given the index
    fn sample(&self, index: usize) -> T;
}
