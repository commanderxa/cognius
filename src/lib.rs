pub mod backward;
pub mod nn;
mod op;
pub mod tensor;
mod tensor_data;
pub mod value;
pub mod value_nn;

pub use tensor::Tensor;
pub use value::Value;
pub use value_nn::MLP;
