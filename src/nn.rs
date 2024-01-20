pub mod criterions;
pub mod functional;
pub mod linear;
pub mod optim;

// define short paths
// layers
pub use linear::Linear;
// criterions
pub use criterions::MSELoss;
