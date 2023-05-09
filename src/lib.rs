pub mod nn;
pub mod value_nn;
mod op;
pub mod tensor;
mod tensor_data;
pub mod value;

pub use value_nn::MLP;
pub use tensor::Tensor;
pub use value::Value;
