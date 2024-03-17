pub mod sgd;

// Short paths for algorithms
pub use sgd::SGD;

/// Behavior of optimizers
pub trait Optim {
    /// Updates the parameters according to the gradients.
    fn step(&self);

    /// Sets gradients to zero.
    fn zero_grad(&self);
}
